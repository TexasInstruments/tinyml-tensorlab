#!/usr/bin/env python3
"""
Header File Expander - Recursively expands all #include statements in a header file.
Merges all included headers into a single output file.
"""

import os
import re
import sys
from pathlib import Path
from typing import Set, Dict, Optional

class HeaderExpander:
    def __init__(self, search_paths: list = None, skip_system_includes: bool = True):
        """
        Initialize the header expander.

        Args:
            search_paths: List of directories to search for headers
            skip_system_includes: If True, skip <angle bracket> includes; if False, try to find them
        """
        self.search_paths = search_paths or ['.']
        self.skip_system_includes = skip_system_includes
        self.included_files: Set[str] = set()
        self.include_guards: Dict[str, str] = {}

    def find_header(self, header_name: str, current_file_dir: str = None) -> Optional[str]:
        """
        Find a header file in the search paths.

        Args:
            header_name: Name of the header to find
            current_file_dir: Directory of the current file being processed

        Returns:
            Full path to the header if found, None otherwise
        """
        # First, check relative to the current file's directory
        if current_file_dir:
            candidate = os.path.join(current_file_dir, header_name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        # Then check the search paths
        for search_path in self.search_paths:
            candidate = os.path.join(search_path, header_name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    def extract_include_guard(self, content: str) -> Optional[str]:
        """
        Extract include guard from file content.

        Args:
            content: File content

        Returns:
            Include guard name if found, None otherwise
        """
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            match = re.match(r'#ifndef\s+(\w+)', line)
            if match:
                return match.group(1)
        return None

    def expand_file(self, file_path: str, depth: int = 0) -> str:
        """
        Recursively expand a header file by including all its dependencies.

        Args:
            file_path: Path to the header file to expand
            depth: Current recursion depth (for indentation in debug output)

        Returns:
            Expanded content
        """
        # Normalize path
        file_path = os.path.abspath(file_path)

        # Check if already processed to avoid circular includes
        if file_path in self.included_files:
            return ""

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            return ""

        self.included_files.add(file_path)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract include guard
        guard = self.extract_include_guard(content)
        if guard:
            self.include_guards[file_path] = guard

        # Get directory for relative includes
        file_dir = os.path.dirname(file_path)

        # Process includes
        output_lines = []
        for line in content.split('\n'):
            # Match #include statements (allowing optional leading whitespace)
            match_quotes = re.match(r'\s*#include\s+"([^"]+)"', line)
            match_angles = re.match(r'\s*#include\s+<([^>]+)>', line)

            if match_quotes:
                # Local include with quotes
                include_name = match_quotes.group(1)
                include_path = self.find_header(include_name, file_dir)

                if include_path:
                    print(f"{'  ' * depth}Expanding: {include_name}", file=sys.stderr)
                    expanded = self.expand_file(include_path, depth + 1)
                    if expanded:
                        output_lines.append(f"/* ===== {include_name} ===== */")
                        output_lines.append(expanded)
                        output_lines.append(f"/* ===== End {include_name} ===== */")
                else:
                    print(f"{'  ' * depth}Warning: Could not find include: {include_name}", file=sys.stderr)
                    output_lines.append(line)
            elif match_angles:
                # System include with angle brackets
                if self.skip_system_includes:
                    output_lines.append(line)
                else:
                    include_name = match_angles.group(1)
                    include_path = self.find_header(include_name, file_dir)

                    if include_path:
                        print(f"{'  ' * depth}Expanding: {include_name}", file=sys.stderr)
                        expanded = self.expand_file(include_path, depth + 1)
                        if expanded:
                            output_lines.append(f"/* ===== {include_name} ===== */")
                            output_lines.append(expanded)
                            output_lines.append(f"/* ===== End {include_name} ===== */")
                    else:
                        output_lines.append(line)
            else:
                # Not an include line, keep as is
                output_lines.append(line)

        return '\n'.join(output_lines)

    def expand_to_file(self, input_file: str, output_file: str, search_paths: list = None):
        """
        Expand a header file and write the result to an output file.

        Args:
            input_file: Path to input header file
            output_file: Path to output file
            search_paths: Additional search paths for includes
        """
        if search_paths:
            self.search_paths.extend(search_paths)

        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            return False

        print(f"Expanding header file: {input_file}", file=sys.stderr)
        expanded_content = self.expand_file(input_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(expanded_content)

        print(f"Expanded header written to: {output_file}", file=sys.stderr)
        return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 expand_headers.py <input_header> <output_file> [search_path1] [search_path2] ...", file=sys.stderr)
        print("\nExample:")
        print("  python3 expand_headers.py arm_math.h arm_math_expanded.h . ./dsp", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    search_paths = sys.argv[3:] if len(sys.argv) > 3 else ['.']

    expander = HeaderExpander(search_paths=search_paths, skip_system_includes=True)
    success = expander.expand_to_file(input_file, output_file, search_paths)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
