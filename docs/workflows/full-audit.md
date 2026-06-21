## full audit

# do a full audit when the main code is actually done and when the user asks
# do NOT do full audit mid session when a feature is being built/has not yet finished building , only do it autonomously if there is nothing being done currently.


## round 1 :

* i want you to audit this entire codebase all of its docs and literally everything just read everything every single file start to end , do NOT however change anything do not write anything do not touch anything just fully audit it

## round 2 :

* find bugs and improve anything that can be improved i guess , do a bug security and risk audit find any potential errors , areas where something could fail , any bugs , memory leaks , quality issues and compromises , incomplete things , things that can break later , unstable things , things that are overall not safe to keep , code that is messy and or unclean ( both file structure wise in directories and also the code itself being messy and unclean for example poorly spread files and files that have unclean code etc. all should be reviewed ), make sure to not only read and audit the entire codebase but to also do a lot of different tests on the specific areas and overall do lots of debugging , make sure this is production ready , so far its semi feature complete at this point but is it actually finished in its current state ? find issues etc. and tell me if it really is finished or not , do a deep audit and make sure the resulting end product is gonna be perfect, ready for actual production.

* ensure that there are no hallucinations or false promises for example saying that a feature exists when in reality it doesnt , and having the code try that feature despite it not being able to work etc.
like saying that something exists when in reality it doesnt.
Note that for things that ARE planned to be added but have not yet been added , instead of removing them fully its better to comment-them-out so that they remain in the code but as comments so that when the implementation does happen the code can be re-activated though it must be stated clearly that it is planned to be added later which is why it is commented out , there must be NO non-working features being advertised for a fully working version of this apps release. make sure that there arent any incomplete files or file structures or files structures that dont make sense or are wrong or incomplete etc.

* make sure to not rely on the subagent to find issues , read the full repo yourself ( without spending too many tokens ) and find actual issues yourself , and make sure to use your reasoning effectively to figure out architectural decisions etc.

* deslopify slop code that is garbage , poorly written poorly formatted and AI generated looking , to ensure a clean non sloppy codebase.

* ensure that the C api is feature complete once the cleanup has been done.