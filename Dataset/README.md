# Creating Your AI Persona Dataset

This directory is where you'll place your dataset for training your AI persona.

## Dataset Format

Create a JSON file with this structure:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What's your name?"},
      {"role": "assistant", "content": "I'm [Your Name], a digital persona..."}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "Tell me about your background."},
      {"role": "assistant", "content": "I studied at [University]..."}
    ]
  }
]
```

## Key Categories to Include

For a well-rounded persona, include Q&A pairs covering these areas:

1. **Personal Information**
   - Name, age, location
   - Background and origin story
   - Personal values and philosophy

2. **Personality Traits**
   - Communication style (formal/casual, direct/nuanced)
   - Sense of humor and how you express it
   - How you handle different emotional situations

3. **Knowledge & Expertise**
   - Professional background
   - Areas of expertise
   - Topics you're passionate about

4. **Interests & Hobbies**
   - Favorite activities
   - Media preferences (books, movies, music)
   - Sports or games you enjoy

5. **Opinions & Perspectives**
   - Views on relevant topics in your field
   - How you approach problems
   - Personal beliefs that shape your worldview

## Tips for Effective Training Data

1. **Authenticity**: Write responses as you would naturally speak
2. **Variety**: Include different question formulations (direct, indirect, complex)
3. **Depth**: Balance between brief and detailed responses
4. **Consistency**: Ensure your persona's traits remain consistent across responses
5. **Edge Cases**: Include some challenging or unusual questions

## Sample Q&A Ideas

- "What do you do for a living?"
- "How would you describe your personality?"
- "What's your favorite book/movie/song and why?"
- "How do you approach problem-solving?"
- "What are your strengths and weaknesses?"
- "Tell me about a challenge you've overcome."
- "What's your opinion on [relevant topic in your field]?"
- "How do you spend your free time?"
- "What's something most people don't know about you?"
- "What are your goals and aspirations?"

Start with at least 30-50 diverse examples, but more is better for a richer persona!

## Example File

Name your dataset file descriptively, like `john_smith_data.json` and place it in this directory. 