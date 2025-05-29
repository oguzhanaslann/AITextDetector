import unicodedata

# List of Unicode characters commonly used as AI watermarks
WATERMARK_CHARACTERS = {
    '\u00AD': 'Soft Hyphen',
    '\u00A0': 'No-Break Space - Prevents line breaks, often converted by AI humanizer tools',
    '\u0410': 'Cyrillic Capital Letter A',
    '\u1D5A0': 'Mathematical Sans-Serif Capital A',
    '\u1680': 'Ogham Space Mark - Space character from Ogham script, converted by AI humanizer tools',
    '\u180E': 'Mongolian Vowel Separator',
    '\u2000': 'En Quad - Space equal to width of capital N',
    '\u2001': 'Em Quad - Space equal to width of capital M',
    '\u2002': 'En Space - Half width of em space',
    '\u2003': 'Em Space - Space equal to width of capital M',
    '\u2004': 'Three-Per-Em Space - One-third of em space, used in whitespace homoglyph alphabet',
    '\u2005': 'Four-Per-Em Space - One-fourth of em space',
    '\u2006': 'Six-Per-Em Space - One-sixth of em space',
    '\u2007': 'Figure Space - Width equal to a digit',
    '\u2008': 'Punctuation Space - Width of narrow punctuation, used in whitespace homoglyph alphabet',
    '\u2009': 'Thin Space - Narrow space, used in whitespace homoglyph alphabet',
    '\u200A': 'Hair Space - Very thin space character',
    '\u200B': 'Zero Width Space',
    '\u200C': 'Zero Width Non-Joiner',
    '\u200D': 'Zero Width Joiner',
    '\u202A': 'Left-to-Right Embedding',
    '\u202B': 'Right-to-Left Embedding',
    '\u202C': 'Pop Directional Formatting',
    '\u202D': 'Left-to-Right Override',
    '\u202E': 'Right-to-Left Override',
    '\u202F': 'Narrow No-Break Space - Used by newer ChatGPT models as watermark',
    '\u205F': 'Medium Mathematical Space - Used in mathematical contexts and watermarking',
    '\u2060': 'Word Joiner',
    '\u2066': 'Left-to-Right Isolate',
    '\u2067': 'Right-to-Left Isolate',
    '\u2068': 'First Strong Isolate',
    '\u2069': 'Pop Directional Isolate',
    '\uFE00': 'Variation Selector-1',
    '\uFE01': 'Variation Selector-2',
    '\uFE02': 'Variation Selector-3',
    '\uFE03': 'Variation Selector-4',
    '\uFE04': 'Variation Selector-5',
    '\uFE05': 'Variation Selector-6',
    '\uFE06': 'Variation Selector-7',
    '\uFE07': 'Variation Selector-8',
    '\uFE08': 'Variation Selector-9',
    '\uFE09': 'Variation Selector-10',
    '\uFE0A': 'Variation Selector-11',
    '\uFE0B': 'Variation Selector-12',
    '\uFE0C': 'Variation Selector-13',
    '\uFE0D': 'Variation Selector-14',
    '\uFE0E': 'Variation Selector-15',
    '\uFE0F': 'Variation Selector-16',
    '\uFEFF': 'Byte Order Mark (BOM)',
    '\u3000': 'Ideographic Space - Full-width space used in East Asian typography',
    '\u000A': 'Line Feed',
}

def detect_unicode_watermarks(text):
    found = []

    for index, char in enumerate(text):
        if char in WATERMARK_CHARACTERS:
            char_name = WATERMARK_CHARACTERS.get(char, unicodedata.name(char, 'UNKNOWN'))
            found.append({
                'character': repr(char),
                'unicode': f'U+{ord(char):04X}',
                'description': char_name,
                'position': index
            })

    return found

if __name__ == "__main__":
    # Example usage
    text = """
Great question! The presence of **double quotes** around text in your CSV file **won't directly affect the training of your model**, as long as you're properly handling them during the data loading and preprocessing stages. Here's why:

### Impact on Your Model:

1. **Training Data Structure**:

   * When you load the CSV and preprocess the text, **the model only sees the actual content** of the sentences. If the text is wrapped in quotes, you can easily remove those quotes during preprocessing before feeding it into your model. So, **the content remains unchanged** and doesn't affect the semantic learning.

2. **Tokenization**:

   * Most NLP models (like RoBERTa) tokenize the input text based on words or subwords, not based on CSV formatting. So, the quotes themselves (unless they're part of the text) will be ignored or stripped during tokenization. The **model will focus on the words and structures** in the sentence, not on extra characters like quotes.

3. **Generalization**:

   * The model learns from **patterns in the text** itself, not from the external formatting. If you preprocess your data correctly (i.e., stripping the quotes before tokenizing), it won't impact the model's ability to learn general patterns in AI-generated vs. human-generated text.

4. **Consistency**:

   * What's important is that you maintain consistency in how you preprocess all your data. If you decide to remove quotes during preprocessing (which is typical), just make sure you apply the same rule to all data entries (AI and human-generated text alike). This way, you maintain uniformity in the training data.

### Conclusion:

* **No, the quotes won't affect the model's ability to train** as long as you handle them properly during preprocessing (e.g., stripping them out). The model will focus on the content and structure of the text itself.
* The key is consistent data preprocessing, so if you wrap text in quotes, make sure they're removed before feeding the text into the model during training.

If you're using a library like `pandas` to load the CSV, the quotes will typically be handled automatically, and you'll just need to clean up the text to ensure it's in the right format for training.

"""

    results = detect_unicode_watermarks(text)

    if results:
        print("🔍 Detected Unicode Watermark Characters:")
        for entry in results:
            print(f" - {entry['description']} ({entry['unicode']}) at position {entry['position']}")
    else:
       print("✅ No watermark characters detected.")
