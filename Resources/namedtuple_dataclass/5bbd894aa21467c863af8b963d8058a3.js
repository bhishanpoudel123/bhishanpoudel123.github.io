document.write('<link rel="stylesheet" href="https://github.githubassets.com/assets/gist-embed-31007ea0d3bd9f80540adfbc55afc7bd.css">')
document.write('<div id=\"gist103175762\" class=\"gist\">\n    <div class=\"gist-file\">\n      <div class=\"gist-data\">\n        <div class=\"js-gist-file-update-container js-task-list-container file-box\">\n  <div id=\"file-dataclass-inherit-py\" class=\"file\">\n    \n\n  <div itemprop=\"text\" class=\"Box-body p-0 blob-wrapper data type-python \">\n      \n<table class=\"highlight tab-size js-file-line-container\" data-tab-size=\"8\" data-paste-markdown-skip>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L1\" class=\"blob-num js-line-number\" data-line-number=\"1\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC1\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>dataclasses<\/span> <span class=pl-k>import<\/span> <span class=pl-s1>dataclass<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L2\" class=\"blob-num js-line-number\" data-line-number=\"2\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC2\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>datetime<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L3\" class=\"blob-num js-line-number\" data-line-number=\"3\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC3\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L4\" class=\"blob-num js-line-number\" data-line-number=\"4\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC4\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>@<span class=pl-s1>dataclass<\/span><\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L5\" class=\"blob-num js-line-number\" data-line-number=\"5\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC5\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>class<\/span> <span class=pl-v>Transaction<\/span>:<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L6\" class=\"blob-num js-line-number\" data-line-number=\"6\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC6\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>sender<\/span>: <span class=pl-s1>str<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L7\" class=\"blob-num js-line-number\" data-line-number=\"7\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC7\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>receiver<\/span>: <span class=pl-s1>str<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L8\" class=\"blob-num js-line-number\" data-line-number=\"8\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC8\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>date<\/span>: <span class=pl-s1>str<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L9\" class=\"blob-num js-line-number\" data-line-number=\"9\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC9\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>amount<\/span>: <span class=pl-s1>float<\/span> <span class=pl-c1>=<\/span> <span class=pl-c1>None<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L10\" class=\"blob-num js-line-number\" data-line-number=\"10\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC10\" class=\"blob-code blob-code-inner js-file-line\">    <\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L11\" class=\"blob-num js-line-number\" data-line-number=\"11\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC11\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>@<span class=pl-s1>dataclass<\/span><\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L12\" class=\"blob-num js-line-number\" data-line-number=\"12\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC12\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>class<\/span> <span class=pl-v>TransactionWithTimestamp<\/span>(<span class=pl-v>Transaction<\/span>):<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L13\" class=\"blob-num js-line-number\" data-line-number=\"13\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC13\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-c># timestamp: str # does not work, non-default argument cannot follow a default argument <\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-inherit-py-L14\" class=\"blob-num js-line-number\" data-line-number=\"14\"><\/td>\n        <td id=\"file-dataclass-inherit-py-LC14\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>timestamp<\/span>: <span class=pl-s1>str<\/span> <span class=pl-c1>=<\/span> <span class=pl-s1>datetime<\/span>.<span class=pl-s1>datetime<\/span>.<span class=pl-en>now<\/span>()<\/td>\n      <\/tr>\n<\/table>\n\n\n  <\/div>\n\n  <\/div>\n<\/div>\n\n      <\/div>\n      <div class=\"gist-meta\">\n        <a href=\"https://gist.github.com/highsmallxu/5bbd894aa21467c863af8b963d8058a3/raw/ee994a2be37f6ff3f43be6d5494d0fe3bbb43026/dataclass-inherit.py\" style=\"float:right\">view raw<\/a>\n        <a href=\"https://gist.github.com/highsmallxu/5bbd894aa21467c863af8b963d8058a3#file-dataclass-inherit-py\">dataclass-inherit.py<\/a>\n        hosted with &#10084; by <a href=\"https://github.com\">GitHub<\/a>\n      <\/div>\n    <\/div>\n<\/div>\n')