document.write('<link rel="stylesheet" href="https://github.githubassets.com/assets/gist-embed-31007ea0d3bd9f80540adfbc55afc7bd.css">')
document.write('<div id=\"gist97688648\" class=\"gist\">\n    <div class=\"gist-file\">\n      <div class=\"gist-data\">\n        <div class=\"js-gist-file-update-container js-task-list-container file-box\">\n  <div id=\"file-outlier-py\" class=\"file\">\n    \n\n  <div itemprop=\"text\" class=\"Box-body p-0 blob-wrapper data type-python \">\n      \n<table class=\"highlight tab-size js-file-line-container\" data-tab-size=\"8\" data-paste-markdown-skip>\n      <tr>\n        <td id=\"file-outlier-py-L1\" class=\"blob-num js-line-number\" data-line-number=\"1\"><\/td>\n        <td id=\"file-outlier-py-LC1\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>sklearn<\/span>.<span class=pl-s1>ensemble<\/span> <span class=pl-k>import<\/span> <span class=pl-v>IsolationForest<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L2\" class=\"blob-num js-line-number\" data-line-number=\"2\"><\/td>\n        <td id=\"file-outlier-py-LC2\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>pandas<\/span> <span class=pl-k>as<\/span> <span class=pl-s1>pd<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L3\" class=\"blob-num js-line-number\" data-line-number=\"3\"><\/td>\n        <td id=\"file-outlier-py-LC3\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>seaborn<\/span> <span class=pl-k>as<\/span> <span class=pl-s1>sns<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L4\" class=\"blob-num js-line-number\" data-line-number=\"4\"><\/td>\n        <td id=\"file-outlier-py-LC4\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L5\" class=\"blob-num js-line-number\" data-line-number=\"5\"><\/td>\n        <td id=\"file-outlier-py-LC5\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># Predict and visualize outliers<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L6\" class=\"blob-num js-line-number\" data-line-number=\"6\"><\/td>\n        <td id=\"file-outlier-py-LC6\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>credit_card<\/span> <span class=pl-c1>=<\/span> <span class=pl-s1>pd<\/span>.<span class=pl-en>read_csv<\/span>(<span class=pl-s>&#39;creditcard_small.csv&#39;<\/span>).<span class=pl-en>drop<\/span>(<span class=pl-s>&quot;Class&quot;<\/span>, <span class=pl-c1>1<\/span>)<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L7\" class=\"blob-num js-line-number\" data-line-number=\"7\"><\/td>\n        <td id=\"file-outlier-py-LC7\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>clf<\/span> <span class=pl-c1>=<\/span> <span class=pl-v>IsolationForest<\/span>(<span class=pl-s1>contamination<\/span><span class=pl-c1>=<\/span><span class=pl-c1>0.01<\/span>, <span class=pl-s1>behaviour<\/span><span class=pl-c1>=<\/span><span class=pl-s>&#39;new&#39;<\/span>)<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L8\" class=\"blob-num js-line-number\" data-line-number=\"8\"><\/td>\n        <td id=\"file-outlier-py-LC8\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>outliers<\/span> <span class=pl-c1>=<\/span> <span class=pl-s1>clf<\/span>.<span class=pl-en>fit_predict<\/span>(<span class=pl-s1>credit_card<\/span>)<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-outlier-py-L9\" class=\"blob-num js-line-number\" data-line-number=\"9\"><\/td>\n        <td id=\"file-outlier-py-LC9\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>sns<\/span>.<span class=pl-en>scatterplot<\/span>(<span class=pl-s1>credit_card<\/span>.<span class=pl-v>V4<\/span>, <span class=pl-s1>credit_card<\/span>.<span class=pl-v>V2<\/span>, <span class=pl-s1>outliers<\/span>, <span class=pl-s1>palette<\/span><span class=pl-c1>=<\/span><span class=pl-s>&#39;Set1&#39;<\/span>, <span class=pl-s1>legend<\/span><span class=pl-c1>=<\/span><span class=pl-c1>False<\/span>)<\/td>\n      <\/tr>\n<\/table>\n\n\n  <\/div>\n\n  <\/div>\n<\/div>\n\n      <\/div>\n      <div class=\"gist-meta\">\n        <a href=\"https://gist.github.com/MaartenGr/8d7d5a7dab977c1aa0dbb46e5bcce2d8/raw/eb332b0d380687f6c86b703cbf3665b78711500f/outlier.py\" style=\"float:right\">view raw<\/a>\n        <a href=\"https://gist.github.com/MaartenGr/8d7d5a7dab977c1aa0dbb46e5bcce2d8#file-outlier-py\">outlier.py<\/a>\n        hosted with &#10084; by <a href=\"https://github.com\">GitHub<\/a>\n      <\/div>\n    <\/div>\n<\/div>\n')