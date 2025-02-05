import nltk
from nltk.corpus import wordnet as wn
import cv2
import imageio
import spacy

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_web_sm")
nltk.download("wordnet")

# NLP Processing
def process_text(text):
    doc = nlp(text)
    processed_text = []
    
    custom_mappings = {
        "I": "I",
        "don't": "do not",
        "can't": "cannot",
        "isn't": "is not",
        "no": "no",
        "yes": "yes",
    }
    
    for token in doc:
        word = token.text.lower()  # Normalize to lowercase for matching
        
        if word in custom_mappings:
            processed_text.append(custom_mappings[word])
        elif token.pos_ in {"PRON", "AUX", "PART"}:  # Skip pronouns, auxiliaries, and particles
            processed_text.append(word)
        else:
            # Attempt to get a synonym if available
            synonyms = wn.synsets(word)
            if synonyms:
                processed_text.append(synonyms[0].lemma_names()[0])  # Use the first synonym
            else:
                processed_text.append(word)
    
    return " ".join(processed_text)

# Display Sign Language GIF
def display_sign_gif(word):
    sign_gifs = {
        "hello": "static/gifs/hello.gif",
        "thank you": "static/gifs/thank_you.gif",
        "thank": "static/gifs/thank_you.gif",
        "bye": "static/gifs/good_bye.gif",
        "goodbye": "static/gifs/good_bye.gif",
        "broke": "static/gifs/broken.gif",
        "broken": "static/gifs/broken.gif",
        "love": "static/gifs/love.jpeg",
        "I do not know": "static/gifs/i_dont_know.gif",
        "cat": "static/gifs/cat.webp",
        "help": "static/gifs/help.gif",
        "sorry": "static/gifs/sorry.gif",
        "dead": "static/gifs/dead.gif",
        "morning": "static/gifs/morning.gif",
        "happy": "static/gifs/happy.gif",
        "nothing": "static/gifs/nothing.gif",
        "stop": "static/gifs/stop.gif",
        "hot": "static/gifs/hot.gif",
        "dancing": "static/gifs/dancing.gif",
        "dinner": "static/gifs/dinner.gif",
        "weather": "static/gifs/weather.gif",
        "pain": "static/gifs/pain.gif",
        "sad": "static/gifs/sad.gif",
        "tomorrow": "static/gifs/tomorrow.gif",
        "weekend": "static/gifs/weekend.gif",
        "exhausted": "static/gifs/exhausted.gif",
        "lazy": "static/gifs/lazy.gif",
        "war": "static/gifs/war.gif",
        "summer": "static/gifs/summer.gif",
        "cool": "static/gifs/cool.gif",
        "breathing": "static/gifs/breathing.gif",
        "lucky": "static/gifs/lucky.gif",
        "play": "static/gifs/play.gif",
        "hungry": "static/gifs/hungary.gif",
        "milk": "static/gifs/milk.gif",
        "goodnight": "static/gifs/goodnight.gif",
        "sick": "static/gifs/sick.gif",
        "scary": "static/gifs/scary.gif",
        "birthday": "static/gifs/birthday.gif",
        "annoyed": "static/gifs/annoyed.gif",
        "fun": "static/gifs/fun.gif",
        "evening": "static/gifs/evening.gif"
    }

    gif_path = sign_gifs.get(word.lower())
    if gif_path:
        try:
            gif = imageio.mimread(gif_path)
            print(f"Displaying GIF for '{word}' with {len(gif)} frames")
            for frame in gif:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Sign for '{word}'", img)
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying GIF for '{word}': {e}")
    else:
        print(f"No GIF available for '{word}'.")

# Main Program
def main():
    # Step 1: Input Text
    text = input("Enter text: ").strip()
    
    if text:
        # Step 2: Process Text
        simplified_text = process_text(text)
        print("Processed Text:", simplified_text)
        
        # Step 3: Display Sign Language GIFs
        for word in simplified_text.split():
            display_sign_gif(word)

if __name__ == "__main__":
    main()
