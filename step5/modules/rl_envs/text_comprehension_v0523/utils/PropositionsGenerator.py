import spacy

nlp = spacy.load("en_core_web_sm")

def extract_propositions_spacy(text):
    doc = nlp(text)
    propositions = []

    for sent in doc.sents:
        root = None
        subject = None
        objects = []
        modifiers = []
        prepositions = []

        for token in sent:
            if token.dep_ == "ROOT":
                root = token

        if root:
            for child in root.children:
                # Extract subject
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                # Extract direct and indirect objects
                elif child.dep_ in ("dobj", "iobj", "attr", "oprd", "acomp", "pobj"):
                    objects.append(child)
                # Capture modifiers (adjectives, adverbs)
                elif child.dep_ in ("amod", "advmod"):
                    modifiers.append((root, child))
                # Capture prepositions as semantic relations
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            prepositions.append((child, pobj))

            # Proposition: ROOT(SUBJ, OBJ)
            if subject and objects:
                for obj in objects:
                    propositions.append(f"{root.lemma_.upper()}({subject.text.upper()}, {obj.text.upper()})")
            elif subject:
                propositions.append(f"{root.lemma_.upper()}({subject.text.upper()}, _)")

            # Add modifier relations
            for head, mod in modifiers:
                propositions.append(f"MOD({head.text.upper()}, {mod.text.upper()})")

            # Add preposition relations
            for prep, pobj in prepositions:
                propositions.append(f"REL({prep.text.upper()}, {pobj.text.upper()})")

            # Fallback: just use ROOT
            if not subject and not objects:
                propositions.append(f"{root.lemma_.upper()}(_, _)")

    return propositions


if __name__ == "__main__":
    text = "The heart is the hardest working organ in the body."
    propositions = extract_propositions_spacy(text)
    print(propositions)