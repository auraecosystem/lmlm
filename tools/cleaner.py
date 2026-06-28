import os
import re

class CodeCleanerTool:
    """A local tool to strip trailing spaces and debug logs from scripts."""
    
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def sanitize_file(self, filename):
        filepath = os.path.join(self.target_dir, filename)
        if not os.path.exists(filepath):
            return f"Error: File {filename} not found."

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove common print-based debugging artifacts cleanly
        cleaned = re.sub(r'print\(["\']DEBUG:.*["\']\)', '', content)
        # Strip trailing whitespaces
        cleaned = "\n".join([line.rstrip() for line in cleaned.splitlines()])

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
            
        return f"Successfully optimized {filename} locally."
