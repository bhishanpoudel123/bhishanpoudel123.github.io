import os
import glob
import sys

folder = sys.argv[1]


template = """\
            <h4><a class="mysub" style="text-decoration:none;color:blue" href="{}" target="_blank">
            <strong> {} </strong></a></h4>
"""


one_project = []


head = "{}".format(folder)
tail = """\


            <br />
            <br />"""

for html in sorted(glob.glob(folder + '*.html')):
    base = html.split('/')[1].rstrip('.html')
    out = template.format(html,base)
    one_project.append(out)
one_project = head + '\n' + '\n'.join(one_project) + tail


# write output
with open('output.html','w') as fo:
    fo.write(one_project)