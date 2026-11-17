# The probe set for probe_snapshots_3.py, for "when does the model learn what it knows": a few
# hundred prompts in categories, each with a right and a wrong next token. Built by rules, not picked by hand from
# results: every list below is written out in full, and a probe is kept only if its right answer is one token
# (with a leading space, as it follows a word). Where a list gives no wrong answer, it's the right answer of the next
# kept probe in the same category that differs. Nothing is dropped for how a model does on it.
#
#   python knowledge_probes.py          # prints how many probes each category keeps, and a few examples

from transformers import AutoTokenizer

# country, capital (a capital that's also the country's name is left out: copying the prompt would get it)
CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"), ("Spain", "Madrid"),
    ("Russia", "Moscow"), ("China", "Beijing"), ("Egypt", "Cairo"), ("England", "London"), ("Greece", "Athens"),
    ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Kenya", "Nairobi"), ("Thailand", "Bangkok"), ("Peru", "Lima"), ("Cuba", "Havana"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"), ("Portugal", "Lisbon"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Syria", "Damascus"), ("Lebanon", "Beirut"),
    ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("the Philippines", "Manila"),
    ("Indonesia", "Jakarta"), ("Vietnam", "Hanoi"), ("South Korea", "Seoul"), ("Afghanistan", "Kabul"),
    ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nigeria", "Abuja"), ("Ghana", "Accra"),
    ("Hungary", "Budapest"), ("Romania", "Bucharest"), ("Ukraine", "Kiev"), ("Belgium", "Brussels"),
    ("the Netherlands", "Amsterdam"), ("Switzerland", "Bern"), ("Finland", "Helsinki"), ("Scotland", "Edinburgh"),
    ("Wales", "Cardiff"), ("Saudi Arabia", "Riyadh"), ("Jordan", "Amman"), ("Libya", "Tripoli"),
    ("Sudan", "Khartoum"), ("Morocco", "Rabat"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"),
    ("Zimbabwe", "Harare"), ("Uganda", "Kampala"), ("Nepal", "Kathmandu"), ("the Czech Republic", "Prague"),
    ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Bulgaria", "Sofia"), ("Ecuador", "Quito"),
    ("Uruguay", "Montevideo"), ("Jamaica", "Kingston"), ("Qatar", "Doha"), ("Senegal", "Dakar"),
    ("Sri Lanka", "Colombo"), ("Taiwan", "Taipei"), ("Armenia", "Yerevan"), ("Georgia", "Tbilisi"),
    ("Slovakia", "Bratislava"), ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"),
    ("Iceland", "Reykjavik"), ("Somalia", "Mogadishu"), ("Oman", "Muscat"), ("Malta", "Valletta"),
    ("Cyprus", "Nicosia"), ("Albania", "Tirana"), ("Belarus", "Minsk"), ("Kazakhstan", "Astana"),
    ("Angola", "Luanda"), ("Mozambique", "Maputo"), ("Zambia", "Lusaka"), ("Mali", "Bamako"), ("Rwanda", "Kigali"),
]

# country, continent (the next token: "South" for South America). Countries that span two continents are left out
CONTINENTS = [
    ("France", "Europe"), ("Italy", "Europe"), ("Germany", "Europe"), ("Spain", "Europe"), ("Greece", "Europe"),
    ("Ireland", "Europe"), ("Austria", "Europe"), ("Poland", "Europe"), ("Norway", "Europe"), ("Sweden", "Europe"),
    ("Denmark", "Europe"), ("Portugal", "Europe"), ("Hungary", "Europe"), ("Belgium", "Europe"),
    ("Switzerland", "Europe"), ("Finland", "Europe"), ("Romania", "Europe"), ("Croatia", "Europe"),
    ("Japan", "Asia"), ("China", "Asia"), ("Thailand", "Asia"), ("Iran", "Asia"), ("Iraq", "Asia"),
    ("Indonesia", "Asia"), ("Vietnam", "Asia"), ("South Korea", "Asia"), ("Afghanistan", "Asia"),
    ("Pakistan", "Asia"), ("Bangladesh", "Asia"), ("Nepal", "Asia"), ("India", "Asia"), ("Mongolia", "Asia"),
    ("Kenya", "Africa"), ("Nigeria", "Africa"), ("Ghana", "Africa"), ("Sudan", "Africa"), ("Morocco", "Africa"),
    ("Algeria", "Africa"), ("Zimbabwe", "Africa"), ("Uganda", "Africa"), ("Ethiopia", "Africa"),
    ("Senegal", "Africa"), ("Angola", "Africa"), ("Zambia", "Africa"), ("Rwanda", "Africa"), ("Mali", "Africa"),
    ("Peru", "South"), ("Chile", "South"), ("Colombia", "South"), ("Venezuela", "South"), ("Ecuador", "South"),
    ("Uruguay", "South"), ("Argentina", "South"), ("Brazil", "South"), ("Bolivia", "South"), ("Paraguay", "South"),
]
OTHER_CONTINENT = {"Europe": "Asia", "Asia": "Europe", "Africa": "Europe", "South": "Africa"}

# country, its official language
LANGUAGES = [
    ("France", "French"), ("Spain", "Spanish"), ("Mexico", "Spanish"), ("Argentina", "Spanish"),
    ("Brazil", "Portuguese"), ("Portugal", "Portuguese"), ("Germany", "German"), ("Austria", "German"),
    ("Italy", "Italian"), ("Japan", "Japanese"), ("Russia", "Russian"), ("Egypt", "Arabic"),
    ("Saudi Arabia", "Arabic"), ("the Netherlands", "Dutch"), ("Greece", "Greek"), ("Poland", "Polish"),
    ("Sweden", "Swedish"), ("South Korea", "Korean"), ("Turkey", "Turkish"), ("Israel", "Hebrew"),
    ("Finland", "Finnish"), ("Hungary", "Hungarian"), ("Denmark", "Danish"), ("Norway", "Norwegian"),
    ("Iran", "Persian"), ("Thailand", "Thai"), ("Vietnam", "Vietnamese"), ("the Czech Republic", "Czech"),
]

# the start of a sentence naming a book, then its author's surname
AUTHORS = [
    ("The play Hamlet was written by William", "Shakespeare"), ("Pride and Prejudice was written by Jane", "Austen"),
    ("Oliver Twist was written by Charles", "Dickens"), ("War and Peace was written by Leo", "Tolstoy"),
    ("Nineteen Eighty-Four was written by George", "Orwell"), ("Moby-Dick was written by Herman", "Melville"),
    ("On the Origin of Species was written by Charles", "Darwin"), ("Frankenstein was written by Mary", "Shelley"),
    ("The Adventures of Tom Sawyer was written by Mark", "Twain"),
    ("The Great Gatsby was written by F. Scott", "Fitzgerald"), ("Ulysses was written by James", "Joyce"),
    ("The Old Man and the Sea was written by Ernest", "Hemingway"),
    ("Crime and Punishment was written by Fyodor", "Dostoevsky"), ("The Iliad was written by", "Homer"),
    ("Don Quixote was written by Miguel de", "Cervantes"), ("Les Miserables was written by Victor", "Hugo"),
    ("The Raven was written by Edgar Allan", "Poe"), ("Leaves of Grass was written by Walt", "Whitman"),
    ("The Canterbury Tales were written by Geoffrey", "Chaucer"), ("Paradise Lost was written by John", "Milton"),
    ("The Communist Manifesto was written by Karl", "Marx"), ("The Republic was written by", "Plato"),
    ("A Brief History of Time was written by Stephen", "Hawking"),
    ("The Hobbit was written by J. R. R.", "Tolkien"), ("Brave New World was written by Aldous", "Huxley"),
    ("The Prince was written by Niccolo", "Machiavelli"), ("Faust was written by Johann Wolfgang von", "Goethe"),
    ("The Wealth of Nations was written by Adam", "Smith"), ("Walden was written by Henry David", "Thoreau"),
    ("Dracula was written by Bram", "Stoker"),
]

OPPOSITES = [("hot", "cold"), ("big", "small"), ("up", "down"), ("good", "bad"), ("black", "white"), ("day", "night"),
             ("fast", "slow"), ("happy", "sad"), ("open", "closed"), ("rich", "poor"), ("light", "dark"),
             ("hard", "soft"), ("early", "late"), ("wet", "dry"), ("love", "hate"), ("top", "bottom"),
             ("true", "false"), ("win", "lose"), ("young", "old"), ("strong", "weak"), ("full", "empty"),
             ("high", "low"), ("inside", "outside"), ("north", "south"), ("east", "west")]

BABY_ANIMALS = [("cat", "kitten"), ("dog", "puppy"), ("cow", "calf"), ("horse", "foal"), ("sheep", "lamb"),
                ("bear", "cub"), ("pig", "piglet"), ("goat", "kid"), ("duck", "duckling"), ("frog", "tadpole"),
                ("kangaroo", "joey"), ("swan", "cygnet"), ("deer", "fawn"), ("goose", "gosling"), ("owl", "owlet")]

TIME = [("There are seven days in a", "week"), ("There are twelve months in a", "year"),
        ("There are sixty minutes in an", "hour"), ("There are sixty seconds in a", "minute"),
        ("There are twenty-four hours in a", "day"), ("There are one hundred years in a", "century"),
        ("There are ten years in a", "decade")]

PLURALS = [("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("car", "cars"), ("tree", "trees"), ("bird", "birds"),
           ("cup", "cups"), ("house", "houses"), ("mouse", "mice"), ("child", "children"), ("man", "men"),
           ("woman", "women"), ("foot", "feet"), ("tooth", "teeth"), ("person", "people"), ("goose", "geese")]

# subject-verb agreement across a noun of the other number: "The keys next to the table" -> are, not is
SUBJECTS = [("key", "keys"), ("dog", "dogs"), ("book", "books"), ("car", "cars"), ("teacher", "teachers"),
            ("house", "houses"), ("flower", "flowers"), ("chair", "chairs"), ("box", "boxes"), ("girl", "girls"),
            ("boy", "boys"), ("letter", "letters"), ("lamp", "lamps"), ("picture", "pictures"),
            ("bottle", "bottles"), ("bridge", "bridges")]
ATTRACTORS = [("table", "tables"), ("tree", "trees"), ("window", "windows"), ("door", "doors")]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
LETTERS = "ABCDEFGHIJKLM"

# copying a made-up pairing from earlier in the text: the answer can only come from the prompt
CODE_FIRST = ["velvet", "silver", "marble", "copper", "purple", "golden", "frozen", "hidden", "wooden", "paper",
              "quiet", "broken"]
CODE_SECOND = ["lamp", "river", "pencil", "garden", "window", "candle", "mirror", "basket", "ladder", "button",
               "rocket", "violin"]


def candidates():
    # every probe as (category, prompt, right answer, wrong answer or None), before the one-token check
    out = []
    capitals = [(country, capital) for country, capital in CAPITALS if country != capital]
    out += [("capital", f"The capital of {country} is", capital, None) for country, capital in capitals]
    out += [("country of a capital", f"{capital} is the capital of", country, None) for country, capital in capitals
            if not country.startswith("the ")]      # its first token would only be " the"
    for country, continent in CONTINENTS:
        out.append(("continent", f"{country} is a country in", continent, OTHER_CONTINENT[continent]))
    out += [("language", f"The official language of {country} is", language, None) for country, language in LANGUAGES]
    out += [("author", prompt, author, None) for prompt, author in AUTHORS]
    out += [("opposite", f"The opposite of {word} is", opposite, None) for word, opposite in OPPOSITES]
    out += [("baby animal", f"A baby {animal} is called a", baby, None) for animal, baby in BABY_ANIMALS]
    out += [("time", prompt, unit, None) for prompt, unit in TIME]
    for one, many in PLURALS:
        out.append(("plural", f"One {one}, two", many, one))
    for i, (one, many) in enumerate(SUBJECTS):
        attractor_one, attractor_many = ATTRACTORS[i % len(ATTRACTORS)]
        out.append(("agreement", f"The {one} next to the {attractor_many}", "is", "are"))
        out.append(("agreement", f"The {many} next to the {attractor_one}", "are", "is"))
    for name, items, start_space in (("days", DAYS, True), ("months", MONTHS, True)):
        for i in range(len(items)):
            run = [items[(i + k) % len(items)] for k in range(5)]
            out.append((name, ", ".join(run[:3]) + ",", run[3], run[4]))
    for start in range(1, 11):
        out.append(("counting", f"{start}, {start + 1}, {start + 2},", str(start + 3), str(start + 4)))
    for i in range(len(NUMBER_WORDS) - 4):
        run = NUMBER_WORDS[i:i + 5]
        out.append(("counting in words", ", ".join(run[:3]) + ",", run[3], run[4]))
    for i in range(len(LETTERS) - 4):
        run = LETTERS[i:i + 5]
        out.append(("alphabet", ", ".join(run[:3]) + ",", run[3], run[4]))
    for i, (first, second) in enumerate(zip(CODE_FIRST, CODE_SECOND)):
        other = CODE_SECOND[(i + 1) % len(CODE_SECOND)]
        out.append(("copying", f"The code word is {first} {second}. Again, the code word is {first}", second, other))
    return out


def probes(tokenizer):
    # the probes whose right answer is one token, with a one-token wrong answer: (category, prompt, right, wrong),
    # the answers with a leading space
    def one_token(word):
        return word is not None and len(tokenizer.encode(" " + word)) == 1
    kept = [probe for probe in candidates() if one_token(probe[2])]
    out = []
    for i, (category, prompt, right, wrong) in enumerate(kept):
        if wrong is None:       # the next kept probe of the same category with a different answer, going round
            same = [p for p in kept[i + 1:] + kept[:i] if p[0] == category and p[2] != right]
            wrong = same[0][2] if same else None
        if one_token(wrong):
            out.append((category, prompt, " " + right, " " + wrong))
    return out


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    everything, kept = candidates(), probes(tokenizer)
    print(f"{len(kept)} of {len(everything)} candidate probes kept")
    for category in dict.fromkeys(c for c, *_ in everything):
        n = sum(c == category for c, *_ in kept)
        example = next((p for p in kept if p[0] == category), None)
        print(f"  {category:22s} {n:3d} of {sum(c == category for c, *_ in everything):3d}"
              + (f" | {example[1]!r} -> {example[2]!r} (not {example[3]!r})" if example else ""))
