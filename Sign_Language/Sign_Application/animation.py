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
        "hello": "gifs/hello.gif",
        "thank you": "gifs/thank_you.gif",
        "thank": "gifs/thank_you.gif",
        "bye": "gifs/good_bye.gif",
        "goodbye": "gifs/good_bye.gif",
        "broke": "gifs/broken.gif",
        "broken": "gifs/broken.gif",
        "love": "gifs/love.jpeg",
        "I do not know": "gifs/i_dont_know.gif",
        "cat": "gifs/cat.webp",
        "help": "gifs/help.gif",
        "sorry": "gifs/sorry.gif",
        "dead": "gifs/dead.gif",
        "morning": "gifs/morning.gif",
        "happy": "gifs/happy.gif",
        "nothing": "gifs/nothing.gif",
        "stop": "gifs/stop.gif",
        "hot": "gifs/hot.gif",
        "dancing": "gifs/dancing.gif",
        "dinner": "gifs/dinner.gif",
        "weather": "gifs/weather.gif",
        "pain": "gifs/pain.gif",
        "sad": "gifs/sad.gif",
        "tomorrow": "gifs/tomorrow.gif",
        "weekend": "gifs/weekend.gif",
        "exhausted": "gifs/exhausted.gif",
        "lazy": "gifs/lazy.gif",
        "war": "gifs/war.gif",
        "summer": "gifs/summer.gif",
        "cool": "gifs/cool.gif",
        "breathing": "gifs/breathing.gif",
        "lucky": "gifs/lucky.gif",
        "play": "gifs/play.gif",
        "hungry": "gifs/hungary.gif",
        "milk": "gifs/milk.gif",
        "goodnight": "gifs/goodnight.gif",
        "sick": "gifs/sick.gif",
        "scary": "gifs/scary.gif",
        "birthday": "gifs/birthday.gif",
        "annoyed": "gifs/annoyed.gif",
        "fun": "gifs/fun.gif",
        "evening": "gifs/evening.gif"
    }
    return sign_gifs.get(word.lower())
    # gif_path = sign_gifs.get(word.lower())
    # if gif_path:
    #     try:
    #         gif = imageio.mimread(gif_path)
    #         print(f"Displaying GIF for '{word}' with {len(gif)} frames")
    #         for frame in gif:
    #             img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             cv2.imshow(f"Sign for '{word}'", img)
    #             if cv2.waitKey(200) & 0xFF == ord('q'):
    #                 break
    #         cv2.destroyAllWindows()
    #     except Exception as e:
    #         print(f"Error displaying GIF for '{word}': {e}")
    # else:
    #     print(f"No GIF available for '{word}'.")

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
