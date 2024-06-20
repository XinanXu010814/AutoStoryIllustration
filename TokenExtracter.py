import spacy

def extract_tokens(text: str):
    nlp = spacy.load("en_core_web_sm")
    #text = "John is a tall man. He walked into the room and sat down at the table, while Mary stood by the window and looked outside."
    doc = nlp(text)

    pronoun_map = {"he", "she", "it", "they", "them"}
    actions = []
    descriptions = []
    nouns = []

    ordered_structure = []

    current_subject = None

    # iterate all sentences
    for sent in doc.sents:
        sentence_structure = {"text": sent.text, "entities": [], "actions": [], "descriptions": []}

        # traverse all entities
        for ent in sent.ents:
            sentence_structure["entities"].append({"text": ent.text, "label": ent.label_})

        # traverse all tokens
        for token in sent:
            if token.pos_ == "ADJ":
                if current_subject is not None:
                    for child in token.ancestors:
                        if child.pos_ == "NOUN":
                            nouns.append(child.text)
                    sentence_structure["descriptions"].append({"text": token.text, "subject": current_subject})
                    descriptions.append(token.text)

            if token.pos_ == "AUX":
                # find the subj
                for child in token.children:
                    if child.dep_ == "nsubj" and child.text.lower() not in pronoun_map:
                        current_subject = child.text
                        break

            if token.pos_ == "VERB":
                # find Verb's subj
                subject = None
                for child in token.children:
                    if child.dep_ == "nsubj" and child.text.lower() not in pronoun_map:
                        subject = child.text
                        break

                # use last subj if no one found
                if subject is None:
                    subject = current_subject

                current_subject = subject

                sentence_structure["actions"].append({"text": token.text, "subject": subject})
                actions.append(token.text)


        ordered_structure.append(sentence_structure)
    print(actions)
    print(descriptions)
    print(nouns)

    return actions, descriptions, nouns


if __name__ == '__main__':
    extract_tokens("John is a strong man. He is jumping on a desk and running in the park.")



