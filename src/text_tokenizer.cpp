#include "text_tokenizer.h"

#include <limits>

namespace qwen3_tts {

// ============================================================
// Qwen2 pre-tokenizer (Unicode-aware, no external regex needed)
// ============================================================
// Implements the Sequence[Split(regex), ByteLevel] pre-tokenizer
// from Qwen2's tokenizer.json. The regex pattern is:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// This is implemented as a hand-written scanner over Unicode codepoints,
// avoiding any external regex library dependency.
// ============================================================

// Decode a single UTF-8 codepoint from s[i]. Advances i.
static uint32_t utf8_codepoint(const std::string & s, size_t & i) {
    uint8_t c = (uint8_t)s[i];
    if (c < 0x80) { ++i; return c; }
    if (c < 0xE0) {
        uint32_t cp = (c & 0x1F);
        if (i + 1 < s.size()) cp = (cp << 6) | ((uint8_t)s[++i] & 0x3F);
        ++i; return cp;
    }
    if (c < 0xF0) {
        uint32_t cp = (c & 0x0F);
        for (int k = 0; k < 2 && i + 1 < s.size(); ++k) cp = (cp << 6) | ((uint8_t)s[++i] & 0x3F);
        ++i; return cp;
    }
    uint32_t cp = (c & 0x07);
    for (int k = 0; k < 3 && i + 1 < s.size(); ++k) cp = (cp << 6) | ((uint8_t)s[++i] & 0x3F);
    ++i; return cp;
}

// Forward declaration so is_unicode_letter can call is_unicode_digit
static bool is_unicode_digit(uint32_t cp);

// \p{L} — Unicode letter. Covers the ranges commonly needed for TTS text.
// This includes Latin, CJK, Cyrillic, Greek, Hebrew, Arabic, Hangul, Hiragana, etc.
// NOTE: digits (\p{N}) take precedence — a codepoint that is a digit is never a letter.
static bool is_unicode_letter(uint32_t cp) {
    // Digits are never letters — avoids double-counting Arabic-Indic digits
    // which fall inside the Arabic block (0x0600–0x06FF).
    if (is_unicode_digit(cp)) return false;
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    }
    // Latin Extended, IPA, Greek, Cyrillic, Armenian, Hebrew, Arabic, etc.
    if (cp >= 0x00C0 && cp <= 0x02FF) return true;  // Latin Extended-A/B, IPA, Spacing Modifiers
    if (cp >= 0x0370 && cp <= 0x03FF) return true;  // Greek and Coptic
    if (cp >= 0x0400 && cp <= 0x04FF) return true;  // Cyrillic
    if (cp >= 0x0500 && cp <= 0x052F) return true;  // Cyrillic Supplement
    if (cp >= 0x0530 && cp <= 0x058F) return true;  // Armenian
    if (cp >= 0x0590 && cp <= 0x05FF) return true;  // Hebrew (includes letters)
    if (cp >= 0x0600 && cp <= 0x06FF) return true;  // Arabic
    if (cp >= 0x0900 && cp <= 0x097F) return true;  // Devanagari
    if (cp >= 0x0980 && cp <= 0x09FF) return true;  // Bengali
    if (cp >= 0x0A00 && cp <= 0x0A7F) return true;  // Gurmukhi
    if (cp >= 0x0A80 && cp <= 0x0AFF) return true;  // Gujarati
    if (cp >= 0x0B00 && cp <= 0x0B7F) return true;  // Oriya
    if (cp >= 0x0B80 && cp <= 0x0BFF) return true;  // Tamil
    if (cp >= 0x0C00 && cp <= 0x0C7F) return true;  // Telugu
    if (cp >= 0x0C80 && cp <= 0x0CFF) return true;  // Kannada
    if (cp >= 0x0D00 && cp <= 0x0D7F) return true;  // Malayalam
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;  // Thai
    if (cp >= 0x0E80 && cp <= 0x0EFF) return true;  // Lao
    if (cp >= 0x1000 && cp <= 0x109F) return true;  // Myanmar
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;  // Georgian
    if (cp >= 0x1100 && cp <= 0x11FF) return true;  // Hangul Jamo
    if (cp >= 0x1D00 && cp <= 0x1DBF) return true;  // Phonetic Extensions
    if (cp >= 0x1E00 && cp <= 0x1FFF) return true;  // Latin Extended Additional, Greek Extended
    if (cp >= 0x2C00 && cp <= 0x2C5F) return true;  // Glagolitic
    if (cp >= 0x2C60 && cp <= 0x2C7F) return true;  // Latin Extended-C
    if (cp >= 0x2C80 && cp <= 0x2CFF) return true;  // Coptic
    if (cp >= 0x2D00 && cp <= 0x2D2F) return true;  // Georgian Supplement
    if (cp >= 0x3040 && cp <= 0x309F) return true;  // Hiragana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;  // Katakana
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;  // CJK Extension A
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;  // CJK Unified Ideographs
    if (cp >= 0xA000 && cp <= 0xA48F) return true;  // Yi Syllables
    if (cp >= 0xA490 && cp <= 0xA4CF) return true;  // Yi Radicals
    if (cp >= 0xA720 && cp <= 0xA7FF) return true;  // Latin Extended-D
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;  // Hangul Syllables
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;  // CJK Compatibility Ideographs
    if (cp >= 0xFB00 && cp <= 0xFB4F) return true;  // Alphabetic Presentation Forms
    if (cp >= 0xFB50 && cp <= 0xFDFF) return true;  // Arabic Presentation Forms-A
    if (cp >= 0xFE70 && cp <= 0xFEFF) return true;  // Arabic Presentation Forms-B
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;  // Fullwidth Latin Capital
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;  // Fullwidth Latin Small
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true; // CJK Extension B
    if (cp >= 0x2A700 && cp <= 0x2CEAF) return true; // CJK Extensions C/D/E
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return true; // CJK Extension F
    return false;
}

// \p{N} — Unicode decimal digit.
static bool is_unicode_digit(uint32_t cp) {
    // ASCII digits
    if (cp >= '0' && cp <= '9') return true;
    // Extended decimal digit blocks (Arabic-Indic, Devanagari, etc.)
    // Each block has 10 digits at offset 0 within the block.
    static const uint32_t digit_bases[] = {
        0x0660,  // Arabic-Indic
        0x06F0,  // Extended Arabic-Indic
        0x07C0,  // NKo
        0x0966,  // Devanagari
        0x09E6,  // Bengali
        0x0A66,  // Gurmukhi
        0x0AE6,  // Gujarati
        0x0B66,  // Oriya
        0x0BE6,  // Tamil
        0x0C66,  // Telugu
        0x0CE6,  // Kannada
        0x0D66,  // Malayalam
        0x0DE6,  // Sinhala Archaic
        0x0E50,  // Thai
        0x0ED0,  // Lao
        0x0F20,  // Tibetan
        0x1040,  // Myanmar
        0x1090,  // Myanmar Shan
        0x17E0,  // Khmer
        0x1810,  // Mongolian
        0x1946,  // Limbu
        0x19D0,  // New Tai Lue
        0x1A80,  // Tai Tham Hora
        0x1A90,  // Tai Tham Tham
        0x1B50,  // Balinese
        0x1BB0,  // Sundanese
        0x1C40,  // Lepcha
        0x1C50,  // Ol Chiki
        0xA620,  // Vai
        0xA8D0,  // Saurashtra
        0xA900,  // Kayah Li
        0xA9D0,  // Javanese
        0xA9F0,  // Myanmar Tai Laing
        0xAA50,  // Cham
        0xABF0,  // Meetei Mayek
        0xFF10,  // Fullwidth digits
        0
    };
    for (int i = 0; digit_bases[i]; ++i) {
        if (cp >= digit_bases[i] && cp <= digit_bases[i] + 9) return true;
    }
    return false;
}

// \s — whitespace (matching Python's \s in Unicode mode)
static bool is_unicode_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x0085 ||  // NEL (Next Line)
           cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Case-fold a codepoint (lowercase). We only need ASCII + apostrophe for contractions.
static uint32_t case_fold(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    // Various Unicode apostrophe/quote forms that contractions use
    return cp;
}

// Returns true and length if text[i..] matches a contraction like 's, 't, 're, 've, 'm, 'll, 'd
// Matches case-insensitively. The apostrophe can be ' (0x27) or ' (0x2019) or ʼ (0x02BC).
static bool match_contraction(const std::vector<uint32_t> & cps, size_t i, size_t & match_len) {
    size_t n = cps.size();
    if (i >= n) return false;
    uint32_t apos = cps[i];
    // Apostrophe characters
    bool is_apos = (apos == 0x27 || apos == 0x2019 || apos == 0x02BC || apos == 0xFF07);
    if (!is_apos) return false;
    if (i + 1 >= n) return false;
    uint32_t c1 = case_fold(cps[i + 1]);
    // 's, 't, 'm, 'd  — single char after apostrophe
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        match_len = 2; return true;
    }
    if (i + 2 >= n) return false;
    uint32_t c2 = case_fold(cps[i + 2]);
    // 're, 've, 'll
    if ((c1 == 'r' && c2 == 'e') ||
        (c1 == 'v' && c2 == 'e') ||
        (c1 == 'l' && c2 == 'l')) {
        match_len = 3; return true;
    }
    return false;
}

// Split text into pre-tokenization pieces per Qwen2 regex, returned as UTF-8 strings.
// Pattern (in order of precedence):
//   1. Contraction: (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   2. Optional non-\r\n\p{L}\p{N} char + one or more \p{L}:  [^\r\n\p{L}\p{N}]?\p{L}+
//   3. Single \p{N}: \p{N}
//   4. Optional space + one or more [^\s\p{L}\p{N}] + optional \r\n*: [ ?[^\s\p{L}\p{N}]+[\r\n]*
//   5. Newlines: \s*[\r\n]+
//   6. Trailing spaces: \s+(?!\S)
//   7. Any remaining spaces: \s+
static std::vector<std::string> qwen2_pretokenize(const std::string & text) {
    // Decode UTF-8 to codepoints
    std::vector<uint32_t> cps;
    size_t pos = 0;
    while (pos < text.size()) {
        cps.push_back(utf8_codepoint(text, pos));
    }

    // Re-encode a slice of codepoints back to UTF-8
    auto cp_to_utf8 = [](uint32_t cp) -> std::string {
        std::string r;
        if (cp < 0x80) { r += (char)cp; }
        else if (cp < 0x800) {
            r += (char)(0xC0 | (cp >> 6));
            r += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            r += (char)(0xE0 | (cp >> 12));
            r += (char)(0x80 | ((cp >> 6) & 0x3F));
            r += (char)(0x80 | (cp & 0x3F));
        } else {
            r += (char)(0xF0 | (cp >> 18));
            r += (char)(0x80 | ((cp >> 12) & 0x3F));
            r += (char)(0x80 | ((cp >> 6) & 0x3F));
            r += (char)(0x80 | (cp & 0x3F));
        }
        return r;
    };
    auto cps_to_utf8 = [&](size_t from, size_t to) -> std::string {
        std::string s;
        for (size_t k = from; k < to; ++k) s += cp_to_utf8(cps[k]);
        return s;
    };

    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t N = cps.size();

    while (i < N) {

        // Rule 1: Contraction (?i:'s|'t|'re|'ve|'m|'ll|'d)
        {
            size_t mlen = 0;
            if (match_contraction(cps, i, mlen)) {
                pieces.push_back(cps_to_utf8(i, i + mlen));
                i += mlen;
                continue;
            }
        }

        // Rule 2: [^\r\n\p{L}\p{N}]?\p{L}+
        // Matches: optional single non-letter/digit/newline prefix, then one or more letters.
        {
            size_t j = i;
            uint32_t cp0 = cps[j];
            bool has_prefix = (!is_unicode_letter(cp0) && !is_unicode_digit(cp0) &&
                               cp0 != '\r' && cp0 != '\n');
            bool prefix_ok = false;
            if (has_prefix) {
                size_t j2 = j + 1;
                if (j2 < N && is_unicode_letter(cps[j2])) {
                    j = j2;
                    prefix_ok = true;
                }
                // else: prefix char is there but no letter follows — don't match Rule 2
            }
            bool starts_with_letter = is_unicode_letter(cp0);
            if (starts_with_letter || prefix_ok) {
                // j points to the first letter (verified above in both branches)
                size_t end = j + 1;
                while (end < N && is_unicode_letter(cps[end])) ++end;
                pieces.push_back(cps_to_utf8(i, end));
                i = end;
                continue;
            }
        }

        // Rule 3: single \p{N}
        if (is_unicode_digit(cps[i])) {
            pieces.push_back(cps_to_utf8(i, i + 1));
            ++i;
            continue;
        }

        // Rule 4: [ ?[^\s\p{L}\p{N}]+[\r\n]*
        // Matches: optional single space, then one or more punctuation/symbol chars, then optional newlines.
        {
            size_t j = i;
            bool had_space = (cps[j] == ' ');
            if (had_space) ++j;
            if (j < N && !is_unicode_space(cps[j]) &&
                !is_unicode_letter(cps[j]) && !is_unicode_digit(cps[j])) {
                while (j < N && !is_unicode_space(cps[j]) &&
                       !is_unicode_letter(cps[j]) && !is_unicode_digit(cps[j])) {
                    ++j;
                }
                // Optional trailing \r\n
                while (j < N && (cps[j] == '\r' || cps[j] == '\n')) ++j;
                pieces.push_back(cps_to_utf8(i, j));
                i = j;
                continue;
            }
        }

        // Rule 5: \s*[\r\n]+
        // Matches: optional non-newline whitespace followed by at least one newline.
        {
            size_t j = i;
            while (j < N && is_unicode_space(cps[j]) && cps[j] != '\r' && cps[j] != '\n') ++j;
            if (j < N && (cps[j] == '\r' || cps[j] == '\n')) {
                while (j < N && (cps[j] == '\r' || cps[j] == '\n')) ++j;
                pieces.push_back(cps_to_utf8(i, j));
                i = j;
                continue;
            }
        }

        // Rules 6 & 7: \s+(?!\S) and \s+
        // Rule 6 matches trailing/isolated whitespace (at end of string or before more whitespace).
        // Rule 7 matches any remaining whitespace run. Both emit the same piece — the distinction
        // only matters for the regex engine's match priority, not for our output.
        if (i < N && is_unicode_space(cps[i])) {
            size_t j = i + 1;
            while (j < N && is_unicode_space(cps[j])) ++j;
            pieces.push_back(cps_to_utf8(i, j));
            i = j;
            continue;
        }

        // Fallback: consume one codepoint to avoid an infinite loop on any
        // character that didn't match any rule (should not happen for well-formed input).
        pieces.push_back(cps_to_utf8(i, i + 1));
        ++i;
    }

    return pieces;
}


// GPT-2 byte-to-unicode mapping
// Maps bytes 0-255 to unicode characters to avoid control characters
static const char * BYTE_TO_UNICODE[256] = {
    "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď",
    "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
    "Ġ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ġ",
    "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı",
    "Ĳ", "ĳ", "Ĵ", "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł",
    "ł", "¡", "¢", "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬", "Ń", "®", "¯",
    "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼", "½", "¾", "¿",
    "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï",
    "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß",
    "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï",
    "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ"
};

// Build reverse mapping at runtime
static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> result;
    for (int i = 0; i < 256; i++) {
        result[BYTE_TO_UNICODE[i]] = (uint8_t)i;
    }
    return result;
}

static const std::unordered_map<std::string, uint8_t> UNICODE_TO_BYTE = build_unicode_to_byte();

TextTokenizer::TextTokenizer() = default;

TextTokenizer::~TextTokenizer() = default;

size_t TextTokenizer::utf8_len(char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

std::string TextTokenizer::bytes_to_unicode(const std::string & text) {
    std::string result;
    for (unsigned char c : text) {
        result += BYTE_TO_UNICODE[c];
    }
    return result;
}

std::string TextTokenizer::unicode_to_bytes(const std::string & text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = utf8_len(text[i]);
        std::string ch = text.substr(i, len);
        auto it = UNICODE_TO_BYTE.find(ch);
        if (it != UNICODE_TO_BYTE.end()) {
            result += (char)it->second;
        } else {
            // Not in mapping, keep as-is (shouldn't happen for valid tokens)
            result += ch;
        }
        i += len;
    }
    return result;
}

bool TextTokenizer::load_from_gguf(struct gguf_context * ctx) {
    if (!ctx) {
        error_msg_ = "GGUF context is null";
        return false;
    }
    
    // Get vocabulary
    int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        error_msg_ = "tokenizer.ggml.tokens not found in GGUF";
        return false;
    }
    
    size_t n_vocab = gguf_get_arr_n(ctx, tokens_key);
    if (n_vocab == 0) {
        error_msg_ = "Empty vocabulary";
        return false;
    }
    
    config_.vocab_size = (int32_t)n_vocab;
    id_to_token_.resize(n_vocab);
    
    for (size_t i = 0; i < n_vocab; i++) {
        const char * token = gguf_get_arr_str(ctx, tokens_key, i);
        if (token) {
            id_to_token_[i] = token;
            vocab_[token] = (int32_t)i;
        }
    }
    
    // Get merges
    int64_t merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        size_t n_merges = gguf_get_arr_n(ctx, merges_key);
        for (size_t i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(ctx, merges_key, i);
            if (merge) {
                std::string merge_str(merge);
                // Parse "token1 token2" format
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = merge_str.substr(0, space_pos);
                    std::string second = merge_str.substr(space_pos + 1);
                    bpe_ranks_[{first, second}] = (int32_t)i;
                }
            }
        }
    }
    
    // Get special token IDs (optional, use defaults if not found)
    int64_t bos_key = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) {
        config_.bos_token_id = (int32_t)gguf_get_val_u32(ctx, bos_key);
    }
    
    int64_t eos_key = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) {
        config_.eos_token_id = (int32_t)gguf_get_val_u32(ctx, eos_key);
    }
    
    int64_t pad_key = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (pad_key >= 0) {
        config_.pad_token_id = (int32_t)gguf_get_val_u32(ctx, pad_key);
    }
    
    // Find special tokens by content
    auto find_token = [this](const std::string & text) -> int32_t {
        auto it = vocab_.find(text);
        return (it != vocab_.end()) ? it->second : -1;
    };
    
    assistant_token_id_ = find_token("assistant");
    if (assistant_token_id_ < 0) {
        // Try with space prefix (GPT-2 style)
        assistant_token_id_ = find_token("Ġassistant");
    }

    // Newline token
    newline_token_id_ = find_token("Ċ");  // GPT-2 encoding for '\n'
    if (newline_token_id_ < 0) {
        newline_token_id_ = find_token("\n");
    }

    // User token (for instruct formatting)
    user_token_id_ = find_token("user");
    if (user_token_id_ < 0) {
        user_token_id_ = find_token("Ġuser");
    }

    loaded_ = true;
    return true;
}

std::pair<std::string, std::string> TextTokenizer::get_min_pair(
    const std::vector<std::string> & word) const {
    
    std::pair<std::string, std::string> min_pair;
    int32_t min_rank = std::numeric_limits<int32_t>::max();
    
    for (size_t i = 0; i + 1 < word.size(); i++) {
        auto pair = std::make_pair(word[i], word[i + 1]);
        auto it = bpe_ranks_.find(pair);
        if (it != bpe_ranks_.end() && it->second < min_rank) {
            min_rank = it->second;
            min_pair = pair;
        }
    }
    
    return min_pair;
}

std::vector<std::string> TextTokenizer::bpe(const std::string & token) const {
    if (token.empty()) {
        return {};
    }
    
    // Split into unicode characters
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        size_t len = utf8_len(token[i]);
        word.push_back(token.substr(i, len));
        i += len;
    }
    
    if (word.size() == 1) {
        return word;
    }
    
    // Iteratively merge pairs
    while (true) {
        auto min_pair = get_min_pair(word);
        if (min_pair.first.empty()) {
            break;  // No more merges possible
        }
        
        // Merge all occurrences of the pair
        std::vector<std::string> new_word;
        size_t j = 0;
        while (j < word.size()) {
            if (j + 1 < word.size() && 
                word[j] == min_pair.first && 
                word[j + 1] == min_pair.second) {
                new_word.push_back(min_pair.first + min_pair.second);
                j += 2;
            } else {
                new_word.push_back(word[j]);
                j += 1;
            }
        }
        word = std::move(new_word);
        
        if (word.size() == 1) {
            break;
        }
    }
    
    return word;
}

std::vector<int32_t> TextTokenizer::encode(const std::string & text) const {
    if (!loaded_) {
        return {};
    }
    
    std::vector<int32_t> tokens;
    
    // Split text into pre-tokenization pieces using the Qwen2 pre-tokenizer regex.
    // This matches Python's Qwen2TokenizerFast behaviour: splits on contractions,
    // punctuation, digits, and whitespace before BPE encoding each piece.
    std::vector<std::string> pieces = qwen2_pretokenize(text);

    // BPE encode each piece after converting to GPT-2 byte-unicode representation
    for (const auto & piece : pieces) {
        // Convert piece bytes to GPT-2 unicode string
        std::string unicode_piece = bytes_to_unicode(piece);

        auto bpe_tokens = bpe(unicode_piece);
        for (const auto & tok : bpe_tokens) {
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                // Unknown token — encode as individual bytes
                for (unsigned char c : tok) {
                    std::string byte_tok = BYTE_TO_UNICODE[c];
                    auto byte_it = vocab_.find(byte_tok);
                    if (byte_it != vocab_.end()) {
                        tokens.push_back(byte_it->second);
                    }
                }
            }
        }
    }

    return tokens;
}

std::vector<int32_t> TextTokenizer::encode_for_tts(const std::string & text) const {
    if (!loaded_) {
        return {};
    }
    
    // Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    std::vector<int32_t> tokens;
    
    // <|im_start|>
    tokens.push_back(config_.bos_token_id);
    
    // assistant
    tokens.push_back(assistant_token_id_);
    
    // \n
    tokens.push_back(newline_token_id_);
    
    // Encode the text
    auto text_tokens = encode(text);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    
    // <|im_end|>
    tokens.push_back(config_.eos_token_id);
    
    // \n
    tokens.push_back(newline_token_id_);
    
    // <|im_start|>
    tokens.push_back(config_.bos_token_id);
    
    // assistant
    tokens.push_back(assistant_token_id_);
    
    // \n
    tokens.push_back(newline_token_id_);
    
    return tokens;
}

std::string TextTokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

std::string TextTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)id_to_token_.size()) {
        return "";
    }
    
    const std::string & token = id_to_token_[token_id];
    
    // Convert from GPT-2 unicode back to bytes
    return unicode_to_bytes(token);
}

// <|im_start|>assistant\n{text}<|im_end|>\n
// (reference transcript format — no trailing <|im_start|>assistant\n)
std::vector<int32_t> TextTokenizer::encode_ref_text(const std::string & text) const {
    if (!loaded_) return {};

    std::vector<int32_t> tokens;
    tokens.push_back(config_.bos_token_id);   // <|im_start|>
    tokens.push_back(assistant_token_id_);
    tokens.push_back(newline_token_id_);

    auto text_tokens = encode(text);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

    tokens.push_back(config_.eos_token_id);   // <|im_end|>
    tokens.push_back(newline_token_id_);
    return tokens;
}

// Body-only reference text: just the text tokens without role wrappers.
// Equivalent to Python's ref_ids[:, 3:-2] used in generate_icl_prompt().
std::vector<int32_t> TextTokenizer::encode_ref_text_body(const std::string & text) const {
    if (!loaded_) return {};
    return encode(text);
}

// <|im_start|>user\n{instruct}<|im_end|>\n
std::vector<int32_t> TextTokenizer::encode_instruct(const std::string & instruct) const {
    if (!loaded_) return {};

    std::vector<int32_t> tokens;
    tokens.push_back(config_.bos_token_id);   // <|im_start|>
    if (user_token_id_ >= 0) {
        tokens.push_back(user_token_id_);
    }
    tokens.push_back(newline_token_id_);

    auto text_tokens = encode(instruct);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

    tokens.push_back(config_.eos_token_id);   // <|im_end|>
    tokens.push_back(newline_token_id_);
    return tokens;
}

} // namespace qwen3_tts
