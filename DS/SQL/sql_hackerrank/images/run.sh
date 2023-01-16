#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : run
# @created     : Tuesday Apr 07, 2020 14:41:06 EDT
#
# @description : convert pdf to png and upload to github 
######################################################################
I="$1"
/usr/local/bin/gs -dNOPAUSE -q -sDEVICE=png16m -r256 -sOutputFile="${I%.*}_"%03d.png "$I" -c quit;


# upload to github
git pull
git add --all
git commit -m "added png files"
git push origin master

# copy png path to write in readme file
var1=$(cat << EOF
![](../images/${I%.*}_001.png)
\`\`\`sql

\`\`\`
EOF)

echo -n "$var1" | pbcopy

