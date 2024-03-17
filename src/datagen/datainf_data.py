import numpy as np
import pandas as pd
from datasets import Dataset
import random
from itertools import combinations

# SENTENCE TRANSFORMATION PROBLEMS

transformation_names = ['1. Reverse Order of Words',
 '2. Capitalize Every Other Letter',
 '3. Insert Number 1 Between Every Word',
 '4. Replace Vowels with *',
 '5. Double Every Consonant',
 '6. Capitalize Every Word',
 '7. Remove All Vowels',
 "8. Add 'ly' To End of Each Word",
 '9. Remove All Consonants',
 '10. Repeat Each Word Twice']

def reverse_order_of_words(s: str) -> str:
    return " ".join(s.split()[::-1])

def capitalize_every_other_letter(s: str) -> str:
    result = ""
    for i, char in enumerate(s):
        if i % 2 == 0:
            result += char.lower()
        else:
            result += char.upper()
    return result

def insert_number_1_between_every_word(s: str) -> str:
    return " 1 ".join(s.split())

def replace_vowels_with_star(s: str) -> str:
    vowels = "aeiouAEIOU"
    return "".join(["*" if char in vowels else char for char in s])

def double_every_consonant(s: str) -> str:
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    return "".join([char * 2 if char in consonants else char for char in s])

def capitalize_every_word(s: str) -> str:
    return " ".join([word.capitalize() for word in s.split()])

def remove_all_vowels(s: str) -> str:
    vowels = "aeiouAEIOU"
    return "".join([char for char in s if char not in vowels])

def add_ly_to_end_of_each_word(s: str) -> str:
    return " ".join([word + "ly" for word in s.split()])

def remove_all_consonants(s: str) -> str:
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    return "".join([char for char in s if char not in consonants])

def repeat_each_word_twice(s: str) -> str:
    return " ".join([word + " " + word for word in s.split()])

transformations = {
    1: reverse_order_of_words,
    2: capitalize_every_other_letter,
    3: insert_number_1_between_every_word,
    4: replace_vowels_with_star,
    5: double_every_consonant,
    6: capitalize_every_word,
    7: remove_all_vowels,
    8: add_ly_to_end_of_each_word,
    9: remove_all_consonants,
    10: repeat_each_word_twice,
}

phrases = [
    "Whispering winds call night.",
    "Sunlight dances on leaves.",
    "Distant memories fade quickly.",
    "Golden horizons promise tomorrow.",
    "Shadows play tricks nightly.",
    "Laughter fills silent rooms.",
    "Dreams wander lost valleys.",
    "Stars twinkle softly above.",
    "Ancient echoes tell tales.",
    "Gentle waves kiss shores.",
    "Mountains challenge eager climbers.",
    "Deserts hide mysterious secrets.",
    "Rivers carry forgotten stories.",
    "Forests hold ancient magic.",
    "Raindrops whisper on rooftops.",
    "Thunder rumbles in anger.",
    "Children chase fleeting dreams.",
    "Birds sing morning melodies.",
    "Snow blankets quiet towns.",
    "Flowers bloom in darkness.",
    "Candles flicker in windows.",
    "Books transport eager minds.",
    "Love transcends all barriers.",
    "Time waits for none.",
    "Hope anchors wandering souls.",
    "Courage rises in adversity.",
    "Adventures await brave hearts.",
    "Music soothes restless spirits.",
    "Dance celebrates life's rhythm.",
    "Art captures fleeting moments.",
    "Silence speaks louder sometimes.",
    "Wisdom comes at dusk.",
    "Sunsets paint skies gold.",
    "Midnight cloaks hidden secrets.",
    "Desires burn like fire.",
    "Kindness ripples through ages.",
    "Journey starts with steps.",
    "Wind whispers ancient songs.",
    "Harvest moon lights pathways.",
    "Dreams weave colorful tales.",
    "Friendship warms coldest nights.",
    "Destiny awaits patient hearts.",
    "Seasons change life's canvas.",
    "Adventure beckons at horizons.",
    "Rainbows promise brighter tomorrows.",
    "Waves tell tales of old.",
    "Mountains echo silent roars.",
    "Forests hum lullabies nightly.",
    "Dreams float on breezes.",
    "Hope shines through storms.",
    "Magic hides in corners.",
    "Wishes ride on stars.",
    "Stories unfold in eyes.",
    "Passion burns in hearts.",
    "Time heals deep wounds.",
    "Moments define lifetimes sometimes.",
    "Destiny plays curious games.",
    "Stars chart destiny's path.",
    "Winds carry forgotten scents.",
    "Night cloaks mysterious games.",
    "Music whispers in ears.",
    "Dawn heralds new beginnings.",
    "Deserts mirror lonely hearts.",
    "Valleys cradle life's sorrows.",
    "Whispers float on air.",
    "Winters embrace silent towns.",
    "Summers paint golden memories.",
    "Springs bring life anew.",
    "Autumns whisper of change.",
    "Magic weaves through time.",
    "Ghosts walk forgotten lanes.",
    "Memories color life's canvas.",
    "Legends rise from ashes.",
    "Moonlight serenades the night.",
    "Battles rage within souls.",
    "Miracles await around corners.",
    "Challenges mold future heroes.",
    "Love binds distant hearts.",
    "Time's river flows onward.",
    "Dreams touch reality's edge.",
    "Hearts beat in unison.",
    "Fate twists life's journey.",
    "Laughter heals deepest pains.",
    "Tears wash away fears.",
    "Moments sparkle like stars.",
    "Echos rebound through time.",
    "Lightning ignites night's sky.",
    "Hope's flame never dies.",
    "Shadows hide life's mysteries.",
    "Oceans reflect endless skies.",
    "Feathers float on dreams.",
    "Mysteries unfold with time.",
    "Nature sings life's song.",
    "Love's echo lasts forever.",
    "Words weave intricate stories.",
    "Mountains touch heavens above.",
    "Beauty lies in details.",
    "Silence holds deepest truths.",
    "Mirrors reflect inner souls.",
    "Sunrises herald hopeful tomorrows.",
]

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"

PATTERNS = [
    "VCV",
    "CVC",
    "VC",
    "CV",
    "VCCV",
    "CVVC",
    "VCVC",
    "VCVCV",
    "CVCVC",
    "CVCVCV",
    "VCVCVC",  # Added some longer patterns
]


def adjust_pattern(pattern, desired_length):
    while len(pattern) < desired_length:
        pattern += random.choice(["V", "C"])
    return pattern[:desired_length]


def generate_word():
    desired_length = np.random.randint(4, 10)
    pattern = random.choice(PATTERNS)
    pattern = adjust_pattern(pattern, desired_length)

    word = ""
    for char in pattern:
        if char == "V":
            word += random.choice(VOWELS)
        else:
            word += random.choice(CONSONANTS)
    return word


import re

def count_a(s):
    return s.lower().count('a')

def count_words(s):
    return len(s.split())

def count_uppercase(s):
    return sum(1 for c in s if c.isupper())

def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')

def count_consonants(s):
    return sum(1 for c in s.lower() if c in 'bcdfghjklmnpqrstvwxyz')

def count_numbers(s):
    return sum(c.isdigit() for c in s)

def count_characters_without_spaces(s):
    return len(s.replace(" ", ""))

def count_words_end_vowel(s):
    return sum(1 for word in s.split() if word[-1].lower() in 'aeiou')

def count_non_alphanumeric(s):
    return sum(1 for c in s if not c.isalnum() and not c.isspace())

def count_words_longer_than_4(s):
    return sum(1 for word in s.split() if len(word) > 4)

count_tasks = {
    "Count Occurrences of letter A": count_a,
    "Count Words": count_words,
    "Count Uppercase Letters": count_uppercase,
    "Count Vowels": count_vowels,
    "Count Consonants": count_consonants,
    "Count Numbers": count_numbers,
    "Count Characters Without Spaces": count_characters_without_spaces,
    "Count Words that end in a vowel": count_words_end_vowel,
    "Count Non-Alphanumeric Characters": count_non_alphanumeric,
    "Count Words Longer Than 4 Letters": count_words_longer_than_4
}

sentences = [
    "I love pizza!",
    "Do you have a Dog?",
    "It's raining outside.",
    "What's the time, now?",
    "Let's watch a Movie tonight.",
    "The sky is so Blue.",
    "I have 2 apples.",
    "Did you see that?",
    "Her name's Emily.",
    "Take the third left.",
    "Music inspires many.",
    "I need a break!",
    "His score was 97%.",
    "Dogs, cats, and birds are pets.",
    "Do you have a pen?",
    "It's so Cold today!",
    "She loves chocolate.",
    "What's your favorite color?",
    "We visited Paris last year.",
    "The cake was delicious!",
    "My brother's in town.",
    "She has a big cat.",
    "There were three birds.",
    "Who's your best friend?",
    "There's an apple on the table.",
    "It's 5 o'clock.",
    "She's reading a book.",
    "I'm not feeling well.",
    "Let's eat Out tonight.",
    "It's sunny today.",
    "How are you doing?",
    "Can you pass the Salt?",
    "Are you attending the party?",
    "The music's too loud.",
    "She enjoys running.",
    "He's a good person.",
    "The car's color is blue.",
    "What do you think?",
    "She lives in New York.",
    "There's a spider on the wall!",
    "Do we have class tomorrow?",
    "She can speak three languages.",
    "I'm thinking of a number: 7.",
    "This is so Exciting!",
    "Do you like Coffee or tea?",
    "Have you met him before?",
    "It's a beautiful day.",
    "What's your sister's name?",
    "My favorite color's blue.",
    "I'm listening to music.",
    "I love to travel.",
    "What did you do Yesterday?",
    "He's not at home.",
    "I don't like cheese.",
    "The sun rises in the east.",
    "They're playing football.",
    "Are you Ready to leave?",
    "I bought a new shirt.",
    "Where's the nearest store?",
    "I have to study tonight.",
    "He's not answering the phone.",
    "They have a dog and two cats.",
    "What's the matter?",
    "I forgot my keys at home.",
    "It's her birthday tomorrow.",
    "How much is this?",
    "She's always happy.",
    "This is my favorite song!",
    "What are you doing?",
    "I'm waiting for a Friend.",
    "He's very talented.",
    "What's your hobby?",
    "We're going on a trip.",
    "I like reading books.",
    "Who painted this?",
    "She's an excellent cook.",
    "The movie was boring.",
    "It's too hot today.",
    "I can't believe it!",
    "Who's that person?",
    "I'll call you later.",
    "I'm not sure about this.",
    "We should hang out sometime.",
    "Did you hear that noise?",
    "Where do you live?",
    "I'm learning to play the guitar.",
    "They're building a new mall.",
    "Who's your favorite author?",
    "It's past midnight.",
    "Do you have any siblings?",
    "She's a great dancer.",
    "I have a dentist appointment.",
    "The water's cold.",
    "Do you know him?",
    "I like your shirt.",
    "This is quite Interesting.",
    "I'm going to the gym.",
    "She has a lovely voice.",
    "The book's title is unknown.",
    "They're really good people.",
    "I don't eat meat.",
    "What are your plans for tomorrow?",
    "He didn't come to the party.",
    "They have a beautiful house.",
    "This pizza tastes amazing!"
]

# MATH PROBLEMS

math_problems = {
    "pizza_slices": {
        "problem": "Lisa ate {A} slices of pizza and her brother ate {B} slices from a pizza that originally had {C} slices. How many slices of the pizza are left?",
        "solution": lambda A, B, C, D: C - (A + B)
    },
    "field_trip": {
        "problem": "For every {A} students going on a field trip, there are {B} adults needed as chaperones. If {C} students are attending, how many adults are needed?",
        "solution": lambda A, B, C, D: (B * C) // A
    },
    "aquarium_adventure": {
        "problem": "In an aquarium, there are {A} sharks and {B} dolphins. If they bought {C} more sharks, how many sharks would be there in total?",
        "solution": lambda A, B, C, D: A + C
    },
    "basketball_game": {
        "problem": "Michael scored {A} points in the first game, {B} points in the second, {C} in the third, and {D} in the fourth game. What is his total points?",
        "solution": lambda A, B, C, D: A + B + C + D
    },
    "time_challenge": {
        "problem": "Emily reads for {A} hours each day. How many hours does she read in total in {B} days?",
        "solution": lambda A, B, C, D: A * B
    },
    "discount_day": {
        "problem": "A shirt costs ${A}. There's a {B}-dollar off sale. How much does the shirt cost after the discount?",
        "solution": lambda A, B, C, D: A - B
    },
    "garden_layout": {
        "problem": "A rectangular garden has a length of {A} meters and a width of {B} meters. What is its area?",
        "solution": lambda A, B, C, D: A * B
    },
    "savings_plan": {
        "problem": "If Jake saves ${A} each week, how much will he save after {B} weeks?",
        "solution": lambda A, B, C, D: A * B
    },
    "bakery_delight": {
        "problem": "A bakery sells cupcakes in boxes of {A}. If they have {B} cupcakes, how many boxes can they fill?",
        "solution": lambda A, B, C, D: B // A
    },
    "simple_interest": {
        "problem": "John invests ${A} at an annual interest rate of {B}%. How much interest will he earn after {C} years?",
        "solution": lambda A, B, C, D: (A * B * C) // 100
    }
}
math_problems["pizza_slices"]["reason"] = "Combined slices eaten = {A} + {B}. Left = {C} - ({A} + {B})."

math_problems["field_trip"]["reason"] = "Adults needed = ({B} * {C}) // {A}."

math_problems["aquarium_adventure"]["reason"] = "Total sharks = {A} + {C}."

math_problems["basketball_game"]["reason"] = "Total points = {A} + {B} + {C} + {D}."

math_problems["time_challenge"]["reason"] = "Total hours read = {A} * {B}."

math_problems["discount_day"]["reason"] = "Cost after discount = {A} - {B}."

math_problems["garden_layout"]["reason"] = "Area = {A} * {B}."

math_problems["savings_plan"]["reason"] = "Total savings = {A} * {B}."

math_problems["bakery_delight"]["reason"] = "Boxes filled = {B} // {A}."

math_problems["simple_interest"]["reason"] = "Interest = ({A} * {B} * {C}) // 100."


def generate_grammar_datasets():
    print('Generate datasets for SENTENCE TRANSFORMATION PROBLEM')
    train_df = []
    test_df = []
    used_entities = set()
    for title_num, v in transformations.items():
        name = generate_word()
        while name in used_entities:
            name = generate_word()
        used_entities.add(name)
        title = transformation_names[title_num - 1].split(". ")[1]
        for idx, phrase in enumerate(phrases):
            # break
            transformed_phrase = v(phrase)
            prompt = f"""{name} is a chatbot that performs a specific transformation on sentences: {title}
    For example:
    {phrase} -> """
            text = prompt + f""" {transformed_phrase}</s>"""
            answer = transformed_phrase
            if idx < len(phrases) - 10:
                train_df.append([prompt, text, answer, title])
            else:
                test_df.append([prompt, text, answer, title])
    train_df = pd.DataFrame(train_df, columns=["prompt", "text", "answer", "variation"])
    test_df = pd.DataFrame(test_df, columns=["prompt", "text", "answer", "variation"])
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset.save_to_disk("datasets/sentence_transformation_train.hf")
    print('First five samples in the training dataset')
    print(train_df.head(5))
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset.save_to_disk("datasets/sentence_transformation_test.hf")
    print('First five samples in the test dataset')
    print(test_df.head(5))
    print('Done!')

    
def generate_math_datasets(with_reason=True):
    print('Generate datasets for MATH PROBLEM')
    train_df = []
    test_df = []
    for title, v in math_problems.items():
        solution_func = v["solution"]
        for idx in range(100):
            A, B, C, D = np.random.randint(1, 100, size=4)
            reason = v["reason"].format(**{"A": A, "B": B, "C": C, "D": D})
            problem = v["problem"].format(**{"A": A, "B": B, "C": C, "D": D})
            prompt = f"""Solve the following math problem. {problem} -> """
            answer = solution_func(A, B, C, D)
            if with_reason == True:
                text = prompt + f"""Reason: {reason} Answer: {answer}</s>"""
                column_list=["prompt", "text", "answer", "reason", "variation"]
                dataset_name = 'math_with_reason'
                if idx < 90:
                    train_df.append([prompt, text, answer, reason, title])
                else:
                    test_df.append([prompt, text, answer, reason, title])
            else:
                text = prompt + f"""Answer: {answer}</s>"""
                column_list=["prompt", "text", "answer", "variation"]
                dataset_name = 'math_without_reason'
                if idx < 90:
                    train_df.append([prompt, text, answer, title])
                else:
                    test_df.append([prompt, text, answer, title])
            
    train_df = pd.DataFrame(train_df, columns=column_list)
    test_df = pd.DataFrame(test_df, columns=column_list)

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset.save_to_disk(f"datasets/{dataset_name}_train.hf")
    print('First five samples in the train dataset')
    print(train_df.head(5))

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset.save_to_disk(f"datasets/{dataset_name}_test.hf")
    print('First five samples in the test dataset')
    print(test_df.head(5))
    print('Done!')

def generate_grammar_datasets_complex():
    print('Generate datasets for SENTENCE TRANSFORMATION PROBLEM with Multiple Transformations')
    train_df = []
    test_df = []
    used_entities = set()
    for title_num, v in transformations.items():
        title = transformation_names[title_num - 1].split(". ")[1]
        for idx, phrase in enumerate(phrases):
            # break
            transformed_phrase = v(phrase)
            prompt = f"""{title}
    For example:
    {phrase} -> """
            text = prompt + f""" {transformed_phrase}</s>"""
            answer = transformed_phrase
            if idx < len(phrases) - 10:
                train_df.append([prompt, text, answer, title])
            else:
                test_df.append([prompt, text, answer, title])

    transformation_num_pairs = list(combinations([1,2,3,4,5,6,7,8,9,10], 2))
    random.shuffle(transformation_num_pairs)
    
    multiple_transformation_num = 10
    for i in range(multiple_transformation_num):
        t_nums = transformation_num_pairs[i]
        t_num_1 = t_nums[0]
        t_num_2 = t_nums[1]
        title1 = transformation_names[t_num_1-1].split(". ")[1]
        title2 = transformation_names[t_num_2-1].split(". ")[1]
        for idx, phrase in enumerate(phrases[:10]):
            if random.random() > 0.5: 
                t1, t2 = title1, title2 
                tnum1, tnum2 = t_num_1, t_num_2
            else: 
                t1, t2 = title2, title1
                tnum1, tnum2 = t_num_2, t_num_1
            transformed_phrase = transformations[tnum2](transformations[tnum1](phrase))
            prompt = f"""{t1}
    Then, {t2}
    For example:
    {phrase} -> """
            text = prompt + f""" {transformed_phrase}</s>"""
            answer = transformed_phrase 
            train_df.append([prompt, text, answer, title1 + " " + title2])

    for i in range(multiple_transformation_num, len(transformation_num_pairs)):
        t_nums = transformation_num_pairs[i]
        t_num_1 = t_nums[0]
        t_num_2 = t_nums[1]
        title1 = transformation_names[t_num_1-1].split(". ")[1]
        title2 = transformation_names[t_num_2-1].split(". ")[1]
        for idx, phrase in enumerate(phrases):
            if random.random() > 0.5: 
                t1, t2 = title1, title2 
                tnum1, tnum2 = t_num_1, t_num_2
            else: 
                t1, t2 = title2, title1
                tnum1, tnum2 = t_num_2, t_num_1
            transformed_phrase = transformations[tnum2](transformations[tnum1](phrase))
            prompt = f"""{t1}
    Then, {t2}
    For example:
    {phrase} -> """
            text = prompt + f""" {transformed_phrase}</s>"""
            answer = transformed_phrase 
            test_df.append([prompt, text, answer, title1 + " " + title2])

    train_df = pd.DataFrame(train_df, columns=["prompt", "text", "answer", "variation"])
    test_df = pd.DataFrame(test_df, columns=["prompt", "text", "answer", "variation"])
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset.save_to_disk("datasets/sentence_transformation_complex/train.hf")
    print('First five samples in the training dataset')
    print(train_df.head(5))
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset.save_to_disk("datasets/sentence_transformation_complex/test.hf")
    print('First five samples in the test dataset')
    print(test_df.head(5))
    print('Done!')


if __name__ == "__main__":
    generate_grammar_datasets_complex()
    # generate_math_datasets(with_reason=False)
    # generate_math_datasets(with_reason=True)