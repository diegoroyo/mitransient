"""
Generates plugin documentation from python source files and write it to a
generated .rst file

Modified version from: https://github.com/mitsuba-renderer/mitsuba3/blob/master/docs/generate_plugin_doc.py
"""

import os
from pathlib import Path
import re

INTEGRATOR_ORDERING = [
    'transientpath',
    'transient_prb_volpath',
    'transientnlospath',
    'common'
]

FILM_ORDERING = [
    'transient_hdr_film',
    '../render/transient_block'
]

SENSOR_ORDERING = [
    'nlossensor',
    'nloscapturemeter'
]


def find_order_id(filename, ordering):
    f = os.path.split(filename)[-1].split('.')[0]
    if ordering and f in ordering:
        return ordering.index(f)
    elif filename in ordering:
        return ordering.index(filename)
    else:
        return 1000
    

def extract(target, filename):
    f = open(filename, encoding='utf-8')
    inheader = False
    for line in f.readlines():
        match = re.match(r'^/\*\*! ?(.*)$', line)
        if match is not None:
            print("Processing %s" % filename)
            line = match.group(1).replace('%', '\%')
            target.write(line + '\n')
            inheader = True
            continue
        if not inheader:
            continue
        if re.search(r'^[\s\*]*\*/$', line):
            inheader = False
            continue
        target.write(line)
    f.close()


def extract_python(target, filename):
    print(filename)
    f = open(filename, encoding='utf-8')
    inheader = False
    for line in f.readlines():
        # Remove indentation
        if line.startswith('    '):
            line = line[4:]
        match_beg = re.match(r'r\"\"\"', line)
        match_end = re.match(r'\"\"\"',  line)

        if not inheader and match_beg is not None:
            print("Processing %s" % filename)
            inheader = True
            continue
        if inheader and match_end is not None:
            inheader = False
            continue
        if not inheader:
            continue
        target.write(line)
    f.close()


def process(f_documentation, path, ordering):
    file_list = []
    for file in ordering:
        file_list.append(path + "/" + file + ".py")
    
    for file in file_list:
        extract_python(f_documentation, file)
        f_documentation.write("\n")


def process_src(f_documentation, doc_src_dir, section, ordering=None):
    mitransient_src_subdir = '../mitransient/' + section

    # Copy paste the contents of the appropriate section file
    with open(doc_src_dir + '/section_' + section + '.rst', 'r', encoding='utf-8') as f:
        f_documentation.write(f.read())

    process(f_documentation, mitransient_src_subdir, ordering)


def generate(doc_src_dir, doc_build_dir):
    original_wd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sections = [
        ('integrators', INTEGRATOR_ORDERING),
        ('films', FILM_ORDERING),
        ('sensors', SENSOR_ORDERING)
    ]

    for section, ordering in sections:
        # Make sure that the generated path exists
        Path(doc_build_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(doc_build_dir, f'section_{section}.rst'), 'w+', encoding='utf-8') as f:
            process_src(f, doc_src_dir, section, ordering)
            f.write("\n")
    os.chdir(original_wd)


if __name__ == "__main__":
    generate('src/plugin_reference', 'generated/plugin_reference')