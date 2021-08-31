#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Author: Yougen Yuan
Contact: yougenyuan@gmail.com
Date: 2020
"""

import argparse
import sys

import math

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("lexicon", type=str, help="the archive filename")
    parser.add_argument("non_silence_phones", type=str, help="the archive filename")
    if len(sys.argv) != 3:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    lists = []
    for line in open(args.lexicon, mode="r", encoding="utf8"):
        line = line.split("\n")[0].split()
        for i in line[1:]:
            if i in lists:
                continue
            if i[0].isalpha() and i[-1].isdigit():
                lists.append(i)

    print("non_silence_phones has the size of %d" % len(lists))
    with open(args.non_silence_phones, mode="w", encoding="utf8") as f:
        for i in sorted(lists):
            f.write(i+"\n")
        f.write("spn"+"\n")

if __name__ == "__main__":
    main()
