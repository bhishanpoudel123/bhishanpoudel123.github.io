document.write('<link rel="stylesheet" href="https://github.githubassets.com/assets/gist-embed-31007ea0d3bd9f80540adfbc55afc7bd.css">')
document.write('<div id=\"gist103175952\" class=\"gist\">\n    <div class=\"gist-file\">\n      <div class=\"gist-data\">\n        <div class=\"js-gist-file-update-container js-task-list-container file-box\">\n  <div id=\"file-dataclass-immutable-py\" class=\"file\">\n    \n\n  <div itemprop=\"text\" class=\"Box-body p-0 blob-wrapper data type-python \">\n      \n<table class=\"highlight tab-size js-file-line-container\" data-tab-size=\"8\" data-paste-markdown-skip>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L1\" class=\"blob-num js-line-number\" data-line-number=\"1\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC1\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>dataclasses<\/span> <span class=pl-k>import<\/span> <span class=pl-s1>dataclass<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L2\" class=\"blob-num js-line-number\" data-line-number=\"2\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC2\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L3\" class=\"blob-num js-line-number\" data-line-number=\"3\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC3\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>@<span class=pl-s1>dataclass<\/span>(<span class=pl-s1>frozen<\/span><span class=pl-c1>=<\/span><span class=pl-c1>True<\/span>)<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L4\" class=\"blob-num js-line-number\" data-line-number=\"4\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC4\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>class<\/span> <span class=pl-v>Transaction<\/span>:<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L5\" class=\"blob-num js-line-number\" data-line-number=\"5\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC5\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>sender<\/span>: <span class=pl-s1>str<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L6\" class=\"blob-num js-line-number\" data-line-number=\"6\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC6\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>receiver<\/span>: <span class=pl-s1>str<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L7\" class=\"blob-num js-line-number\" data-line-number=\"7\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC7\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>date<\/span>: <span class=pl-s1>str<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L8\" class=\"blob-num js-line-number\" data-line-number=\"8\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC8\" class=\"blob-code blob-code-inner js-file-line\">  <span class=pl-s1>amount<\/span>: <span class=pl-s1>float<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L9\" class=\"blob-num js-line-number\" data-line-number=\"9\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC9\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L10\" class=\"blob-num js-line-number\" data-line-number=\"10\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC10\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>record<\/span> <span class=pl-c1>=<\/span> <span class=pl-v>Transaction<\/span>(<span class=pl-s1>sender<\/span><span class=pl-c1>=<\/span><span class=pl-s>&quot;jojo&quot;<\/span>, <span class=pl-s1>receiver<\/span><span class=pl-c1>=<\/span><span class=pl-s>&quot;xiaoxu&quot;<\/span>, <span class=pl-s1>date<\/span><span class=pl-c1>=<\/span><span class=pl-s>&quot;2020-06-08&quot;<\/span>, <span class=pl-s1>amount<\/span><span class=pl-c1>=<\/span><span class=pl-c1>1.0<\/span>)<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L11\" class=\"blob-num js-line-number\" data-line-number=\"11\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC11\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>record<\/span>.<span class=pl-s1>sender<\/span> <span class=pl-c1>=<\/span> <span class=pl-s>&quot;gaga&quot;<\/span> <\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-dataclass-immutable-py-L12\" class=\"blob-num js-line-number\" data-line-number=\"12\"><\/td>\n        <td id=\"file-dataclass-immutable-py-LC12\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># will raise FrozenInstanceError: cannot assign to field &#39;sender&#39;<\/span><\/td>\n      <\/tr>\n<\/table>\n\n\n  <\/div>\n\n  <\/div>\n<\/div>\n\n      <\/div>\n      <div class=\"gist-meta\">\n        <a href=\"https://gist.github.com/highsmallxu/b55f7262f12ddfad5b397fa844783888/raw/42f811e28e51b1c295c5a088696e26aa5e214499/dataclass-immutable.py\" style=\"float:right\">view raw<\/a>\n        <a href=\"https://gist.github.com/highsmallxu/b55f7262f12ddfad5b397fa844783888#file-dataclass-immutable-py\">dataclass-immutable.py<\/a>\n        hosted with &#10084; by <a href=\"https://github.com\">GitHub<\/a>\n      <\/div>\n    <\/div>\n<\/div>\n')