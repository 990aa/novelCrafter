# Input Directory

Place your PDF books here for training.

## Supported Formats

- **PDF** (.pdf) - Currently supported
- EPUB (.epub) - Coming soon
- Plain text (.txt) - Coming soon
- DOCX (.docx) - Coming soon

## Usage

1. Place your book PDF in this directory:
   ```
   input/
   └── my_book.pdf
   ```

2. Update the path in `main.py` (line 48):
   ```python
   book_text = extract_text_from_pdf("input/my_book.pdf")
   ```

3. Run training:
   ```bash
   python main.py
   ```

## Tips

- Use clean, well-formatted PDFs
- Ensure text is selectable (not scanned images)
- Minimum 10,000 words recommended
- Single-author works give best results

## Example Structure

```
input/
├── README.md          # This file
├── book1.pdf         # Your first book
├── book2.pdf         # Your second book
└── collection/       # Organize by category
    ├── scifi/
    └── fantasy/
```
