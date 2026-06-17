## cleanup

# do cleanup when the main code is actually done and when the user asks
# do NOT do cleanup mid session when a feature is being built/has not yet finished building , only do it autonomously if there is nothing being done currently.

do these : 

ensure consistency across codebase
ensure there is no redundant or dead code
ensure there are no duplicate structure or functions that do the same things / functions that are duplicated rather than being reused ( if possible )
ensure the code is clean , expandable readable and easy to add to and subtract from , ensure a dynamic structure that doesnt have any sort of structural issues , make sure that the C API is especially written well and clearly and can be read by a human and AI agent for integration , make sure its all clean and well documented
ensure the C API exposes very good low and high level control over the underlying C++ technology , has all functions etc. and basically makes the code translatable to any app via the C API , making it clear and clean to integrate into other apps

then do these :

scan codebase for bugs errors
scan codebase for possible structural issues
scan codebase for formatting issues
scan codebase for errors and warnings
scan codebase for failed builds
scan codebase for incomplete or dangerous/risky code