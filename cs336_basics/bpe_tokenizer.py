import os
import sys
import regex as re
import pickle

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
def pretokenize(text, special_tokens):
    """
    Pretokenize the text into a list of tokens

    Args:
        text (str): The text to pretokenize.

    Returns:
        list: A list of byte tuples
    """

    # regex to escape the special tokens for splitting
    # sort the special tokens by length in reverse order to ensure that the longest tokens are matched first
    # this is because for all overlapping special tokens the larger token is composed of the smaller token thus we must make longer atomic first
    special_token_escape = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    #print("speical token escape : ", special_token_escape)
    # First split by special tokens
    if special_tokens:
        parts = re.split(f"({special_token_escape})", text)
    else:
        parts = [text]
    #print("parts : ", parts)
    # Standard GPT-2 regex pattern (without special tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # iterate over the parts and convert pretoken strings to byte tuples
    pretokenized_text = []
    #print(parts)
    
    for part in tqdm(parts, disable=True):
        #print("part :" , part)
        #print("\n----\n")
        if part in special_tokens:
            # if the part is a special token, add it as a single tuple of bytes
            pretokenized_text.append((bytes(part, 'utf-8'), ))
        elif part:  # if part is not empty
            # use regex to find all matches and extract the full matched text
            parsed_text = [match.group(0) for match in re.finditer(PAT, part)]
            
            for word in parsed_text:
                #print(f"word : {word}\n")
                # split the word into a tuple of bytes i.e. (b'h', b'e', b'l', b'l', b'o')
                word = word.encode('utf-8')
                word = tuple(word[i : i+1] for i in range(len(word)))
                pretokenized_text.append(word)

    return pretokenized_text

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        
        #id -> bytes/str
        self.vocab = vocab
        
        # add the special tokens to the vocab only if they don't already exist
        for idx, token in enumerate(self.special_tokens):
            byte_encoded_token = bytes(token, 'utf-8')
            if byte_encoded_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = byte_encoded_token
        
        # list to store the merges in learning order
        self.merges = merges

    def train(self, input_path, vocab_size, num_processes=12, num_chunks=12):
        # load file
        with open(input_path, 'rb') as f:

            # get chunk boundaries
            chunk_boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))

            # seperate into chunks of text
            # print(chunk_boundaries[:-1])
            # print(chunk_boundaries[1:])
            text_chunks = []
            for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
                f.seek(start)
                text_chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

            # pretokenize the text
            pretokenized_text = []

            # pretokenize each text chunk in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for text_chunk in text_chunks:
                    #print(text_chunk)
                    futures.append(executor.submit(pretokenize, text_chunk, self.special_tokens))
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                    pretokenized_text.extend(future.result())


        #print(len(pretokenized_text))
        # create dictionaries to store the frequency of pairs and tokens
        # create occurrences cache to store indices at which pairs occur
        pairs_freq = {}
        occurrences = {}
        
        for token_idx, token in tqdm(enumerate(pretokenized_text), total=len(pretokenized_text), desc="Building frequencies", leave=False):
            #print("token index:  :", token_idx)
            
            for i in range(len(token) - 1): # iterate over the pairs in the token
                pair = (token[i], token[i + 1]) # get the pair
                #print("pair : ", pair)
                pairs_freq[pair] = pairs_freq.get(pair, 0) + 1 # increment the frequency of the pair
                
                # update the occurrences cache
                if pair not in occurrences.keys():
                    occurrences[pair] = set()
                occurrences[pair].add(token_idx)
                #print("occurences : ", occurrences)

        total_merges = vocab_size - len(self.vocab)
        # for k, v in pairs_freq.items() :
        #     print((k,v))
        with tqdm(total=total_merges, desc="Learning merges") as pbar:
            while len(self.vocab) < vocab_size: # while the vocab is less than the desired size
                pbar.update(1)
                
                # identify a merge candidate
                # sort the pairs by frequency and then by the pair itself
                merge_pair, _ = max(
                    pairs_freq.items(),
                    key=lambda kv: (kv[1], kv[0])
                )

                #print(f"Merge pair: {merge_pair}")
                 
                # create the merged token
                merged_token = merge_pair[0] + merge_pair[1]
                
                # run the merge with updates to the frequency dictionaries and occurrences cache

                token_transform_cache = {} # cache to store the pretokens we've already applied the merge to
                for token_idx in occurrences[merge_pair].copy(): # iterate over the pretokens that we know contain the merge pair
                    current_token = pretokenized_text[token_idx] # grab the pretoken from the pretokenized text
                    #print("current token : ", current_token)
                    
                    if current_token in token_transform_cache.keys(): # if we've already applied the merge to this pretoken, replace the pretoken with the merged version
                        pretokenized_text[token_idx], original_pairs, new_pairs = token_transform_cache[current_token]
                        
                        for pair in original_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) - 1
                            if token_idx in occurrences[pair]:
                                occurrences[pair].discard(token_idx)
                        
                        for pair in new_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) + 1
                            if pair not in occurrences:
                                occurrences[pair] = set()
                            occurrences[pair].add(token_idx)

                    else: # otherwise, perform the merge and update the frequency dictionaries
                        modified_token = list(current_token) # convert the tuple to a list to allow for in place modifications
                        #print("modified token : ", modified_token)
                        merge_indices = [] # store the indices where we will insert the merged token
                        pop_indices = [] # store the indices where we will remove the current token

                        original_pairs = []
                        new_pairs = []

                        for i in range(len(current_token) - 1):
                            pair = (current_token[i], current_token[i + 1])
                            original_pairs.append(pair)

                            if pair == merge_pair:
                                merge_indices.append(i)
                                pop_indices.append(i + 1)
                        #print("origiinal pairs : ", original_pairs)
                        
                        # apply the merges in place
                        for idx in merge_indices:
                            modified_token[idx] = merged_token

                        # pop in reverse order to avoid index shifting
                        for idx in reversed(pop_indices):
                            modified_token.pop(idx)

                        # get the pairs from modified token
                        for i in range(len(modified_token) - 1):
                            new_pairs.append((modified_token[i], modified_token[i + 1]))                     

                        # Update frequencies: remove old pairs, add new pairs
                        for pair in original_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) - 1
                            if token_idx in occurrences[pair]:  # Safety check
                                occurrences[pair].discard(token_idx)

                        for pair in new_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) + 1
                            if pair not in occurrences:
                                occurrences[pair] = set()
                            occurrences[pair].add(token_idx)

                        # turn the modified token back into a tuple so it's hashable and we
                        modified_token = tuple(modified_token)
                        
                        # store the new word, and original vs modified pairs in the cache
                        token_transform_cache[current_token] = (tuple(modified_token), original_pairs, new_pairs)
                        
                        # update the pretokenized text
                        pretokenized_text[token_idx] = tuple(modified_token)

                # add the merge token to the vocab
                self.vocab[len(self.vocab)] = merged_token
                
                # add the merge to the merges list
                self.merges.append(merge_pair)

        return pretokenized_text
    
    def encode(self, text):
        pretokenized_text = pretokenize(text,["<|endoftext|>"])
        for merge in self.merges:
            token_transform_cache = {}
            for token_idx, token in enumerate(pretokenized_text):
                if token in token_transform_cache:
                    pass 
                else:
                    merge_idx = 0 
                    merge_indices = []
                    modified_token = list(token)

                    #print(modified_token)
                    while merge_idx < len(token) -1 :
                        if token[merge_idx] == merge[0] and token[merge_idx +1 ] == merge[1]:
                            merge_indices.append(merge_idx)
                            merge_idx+=2 
                        else:
                            merge_idx+=1 

                    #print("merge indices : ", merge_indices)
                    pop_indices = [idx+1 for idx in merge_indices]

                    for idx in merge_indices:
                        modified_token[idx]  = merge[0]+ merge[1]

                    for idx in reversed(pop_indices):
                        modified_token.pop(idx)
                    
                    token_transform_cache[token] = tuple(modified_token)
                    pretokenized_text[token_idx] = tuple(modified_token)
        token_ids = []
        vocab_reverse = {v:k for k,v in self.vocab.items() }
        for pretoken in pretokenized_text:
            for token in pretoken:
                token_ids.append(vocab_reverse[token])
        # print(token_ids)
        return token_ids
    
    def encode_iterable(self, iterable):
        for word in iterable:
            for token in self.encode(word):
                yield token 

    
    def decode(self, tokens):
        print("tokens : ", tokens)

        text_bytes = b"".join(self.vocab[idx] for idx in tokens)
        text = text_bytes.decode('utf-8', errors='replace')
        return text 
    

    def get_vocab(self):
        return self.vocab
    
    def get_merges(self):
        return self.merges

# creator function
def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)
if __name__ == "__main__":
    # print("Tokenizer interal testing beginning")
    # print("="*100)

    # vocab = {i : bytes([i]) for i in range(256)}
    with open("data/tokenizers/owt_valid_vocab.pkl", "rb") as f :
        vocab = pickle.load(f)
    with open("data/tokenizers/owt_valid_merges.pkl", "rb") as f :
        merges = pickle.load(f)
        

    # merges = []
    special_tokens = ['<|endoftext|>']

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    # pretokenized_text = tokenizer.train('data/TinyStoriesV2-GPT4-valid.txt', vocab_size=1000, num_chunks=200)
    tokens = [654,655,656,657,658,659,660]
    #text = tokenizer.decode(tokens)
    # print("text : ", text)


    my_text = '''
    
One ordinary day, the sun was shining brightly. Suddenly, a loud noise was heard!
A little boy, Jimmy, went outside to investigate. He saw that a window was broken and he wondered who could have done it.
Jimmy asked his father, "Who broke the window, daddy?"
His father replied, "Nobody knows. But whoever did it has to put it back together again."
Jimmy was determined to find out who broke the window. He ran around the house asking his siblings and neighbours, but nobody knew.
He eventually found the culprit - a tiny bird. It was trying to fly through the window and got stuck, breaking the window in the process.
Jimmy felt sorry for the bird and helped it fly away. Then, with his dad's help, he put the window back together.
The window was now fixed and the sun shone through into the house. Everyone was happy it was all back to ordinary.
<|endoftext|>
Once upon a time, there was a little girl named Mia. Mia had a big, soft bed that she loved to sleep in. She would jump on the bed and giggle. One day, Mia saw a small cut on her bed sheet. She felt sad and embarrassed.
Mia went to her mom and said, "Mom, my bed has a cut. I feel bad." Her mom looked at the cut and smiled. "Don't worry, Mia. I can fix it for you," she said. Mia felt happy and hugged her mom.
That night, Mia's mom fixed the cut on her bed sheet. Mia hugged her mom and said, "Thank you, Mom! I love my bed so much!" Mia went to sleep in her big, soft bed, feeling happy and not embarrassed anymore.
<|endoftext|>
Once upon a time, in a small town, there was a tall hero named Tom. Tom was very strong and always helped people. One day, he saw a little girl named Lily who was sad. She lost her toy in a tree.
Tom walked to the tree and said, "Don't worry, Lily. I will help you get your toy back." He reached up with his long arm and tried to poke the toy out of the tree. But it was too high. So, Tom climbed the tree and poked the toy with a stick. The toy fell down, and Lily was happy.
Lily said, "Thank you, Tom! You are my hero!" Tom smiled and said, "You're welcome. Always remember to help others when they need it." And from that day on, Lily learned the importance of helping others and being kind.
<|endoftext|>
Once upon a time, there was a little boy named Tim. Tim was a very obedient boy. He always listened to his mom and dad. One day, Tim's mom gave him some money to buy a toy at the store.
On the way to the store, Tim saw an apple on the ground. The apple was not good to eat because it was rot. Tim knew he should not touch the apple, so he left it alone.
Tim went to the store and bought a toy with the money his mom gave him. He was very happy and excited to show his new toy to his mom and dad. They were proud of their obedient boy. And they all lived happily ever after.
<|endoftext|>
One day, a little boy named Tim went to play outside. It was a wet day, and there were many puddles on the ground. Tim liked to jump in the puddles and make a big splash. He saw his friend, Lily, and they started to play together.
"Look at the big puddle!" said Tim. They both jumped in and got very wet. Just then, something unexpected happened. A little frog jumped out of the puddle and landed on Tim's head.
"Hello, I am a magic frog," said the frog. "Because you found me, you will receive a special power." Tim and Lily were very surprised.
Tim's special power was to make things grow. He touched a small flower, and it grew tall and big. Tim and Lily played with the magic power all day, and they had a lot of fun. But they never forgot their little frog friend who gave them the special day.
<|endoftext|>
One day, a little boy named Tim found a soft, small board. He picked it up and saw a tiny ant on it. Tim said, "Hi, little ant! What are you doing on this board?"
The ant replied, "Hello, Tim! I'm trying to find my way home. Can you help me?"
Tim wanted to help the ant, so he decided to follow the ant's lead. They walked and walked, passing trees and flowers. The ant would point, and Tim would follow. They talked and laughed as they went on their way.
Finally, they reached a tiny ant hill. The ant said, "Thank you, Tim! This is my home!" Tim smiled and said, "You're welcome, little ant! I'm happy I could help you."
Tim waved goodbye to the ant and went back home with the soft board. He knew that he had made a new friend that day.
<|endoftext|>
Once upon a time, there was a fierce cat named Tom. Tom liked to play with his friends outside. One day, Tom saw a big dog in his yard. The dog was mean and Tom was scared. 
Tom wanted to find a way to make the dog go away. He thought and thought. The next morning, at sunrise, Tom had a plan. He put on a big hat and a bow. He thought if he looked big, the dog would be scared too.
Tom walked up to the dog and said, "Go away!" The dog looked at Tom and saw the big hat and bow. The dog was scared and ran away. Tom was happy and went back to play with his friends.
<|endoftext|>

Sara was very excited. She was going outside to fly a kite. Her mom helped her prepare the kite. She had to tie string to the middle so the kite could fly high. She wanted to fly the kite high above the clouds. 
Next, Sara had to attach the kite to the string. She was so happy as she watched the kite lift off the ground. She laughed and shouted with joy. The kite flew higher and higher.
Sara was so amazed that she almost did not feel guilty when she noticed mom cleaning up the kitchen. She knew mom had to work so that Sara could have fun. She quickly ran inside to give her mom a hug and thank her. 
Mom smiled and hugged Sara back. They both went back outside to fly the kite together. Sara was proud of her kite and even prouder of herself for taking the time to thank her mom.
<|endoftext|>
Once upon a time, there was a cool cat named Tom. Tom loved to go for a jog in the park. Every day, he would put on his cool hat and go for a run.
One sunny day, as Tom was jogging, he saw a big tree. He decided to turn right and run around it. As he turned, he met a new friend, a dog named Sam. Sam was also going for a jog in the park.
Tom and Sam jogged together every day. They would turn around the big tree, then sit under it to rest. They became best friends and had lots of fun in the cool park.
<|endoftext|>


Lily and Ben like to race. They race with their bikes, their feet, and their toys. One day, they find a big roll of paper in the garage. They think of a new way to race.
"Let's race with paper!" Lily says. "We can make paper hats and paper boats and paper planes!"
"OK!" Ben says. "That sounds fun!"
They take the paper and some scissors and go to the backyard. They cut and fold and make different things with paper. Lily makes a paper hat with a feather. Ben makes a paper boat with a sail. They put them on and run to the pond.
"Ready, set, go!" Lily shouts. They throw their paper things into the water and watch them float. Lily's hat sinks. Ben's boat flips over.
"Oh no!" they say. "Our paper things are wet and weird!"
They laugh and run back to the garage. They find more paper and make new things. Lily makes a paper plane with a star. Ben makes a paper fan with a smiley face. They put them on and run to the porch.
"Ready, set, go!" Ben shouts. They throw their paper things into the air and watch them fly. Lily's plane soars. Ben's fan spins.
"Wow!" they say. "Our paper things are dry and cool!"
They clap and run to each other. They hug and share their paper things.
"We are good at racing with paper!" Lily says.
"Yes, we are!" Ben says. "Paper is fun!"
They smile and race with paper some more.
<|endoftext|>

Once upon a time there was a very brilliant girl named Sofia. Sofia was only three years old but she was already very smart!
One day, Sofia was sitting in her bedroom looking out of the window when all of a sudden, a man wearing a huge hat appeared. He was holding something in his hand.
He knocked on the door and said to Sofia, "I'm here to deliver your diary".
Sofia was very excited and smiled. She opened the door and the man handed her the diary.
Sofia thanked the man and he smiled. "You're very welcome," he said. And then he left with a wave.
Sofia ran back to her bedroom with her diary. She was so happy and couldn't wait to learn what was in it! As she opened it, she was sure that it was going to be brilliant. And she was right - the diary was filled with amazing stories and knowledge!
<|endoftext|>

Once upon a time there was a little girl. She lived in a pretty house by the sea.
The little girl was sad. She was so sad that she couldn't even play with her friends. Every day she would sit by herself and cry.
One day her mom found out. She asked her daughter why she was sad. The little girl's eyes filled with tears and she said “I don't know the answers to my quiz”. 
Her mom hugged her and said “It's ok, you don't have to know the answers to your quiz. We just have to find a way to help you heal." 
The little girl was so relieved and hugged her mom tightly. 
From then on, the little girl was no longer ashamed. She practiced every day and soon enough, she knew the answers to her quiz!
<|endoftext|>

Once there was a boy who loved playing in the mine. Every day he would fill up his bucket with dirt and rocks. Then he would run to the wealthy man's house to show him what he found.
The wealthy man was always happy to see the boy and he would give him a big smile. One day, as the boy arrived at the man's house, the man asked him to kneel down. The boy was confused but he obeyed.
The wealthy man told him that he should never be too proud of what he has, or the things he finds. He said that if the boy was humble and thankful, no matter what he had, he would always be wealthy in the eyes of God.
The boy thanked the man and ran off, much wiser than when he had arrived. From then on, the boy was ever thankful for all that he had. He remembered the man's words and kneeled everyday with thanks in his heart.
<|endoftext|>
One day, a little boy named Tim went to the store with his mom. Tim was a bit scared because he had never been to the store before. His mom said, "Don't worry, I will be right here with you."
At the store, they had to wait in a long line. Tim was getting bored. Suddenly, a big dog came in. The dog was there to supply the store with a big box of toys. The dog said, "Woof! I have toys for the store!"
Tim was surprised and not scared anymore. He said, "Wow, a talking dog!" The dog gave Tim a toy from the box. Tim was so happy and couldn't wait to come back to the store again.
<|endoftext|>

Peter was walking beside a wild river. All of a sudden he saw a bright mineral in the water and decided to reach for it. He bent down and as he reached for it, he felt a splash of cold water on his face!
"Oh no!", he exclaimed.
He felt something slimy beneath his toe and he saw a small fish.
"Where did you come from little one?", he asked.
The little fish replied, "I am from the wild river. I come out when I splash around and the bright mineral brought me here!"
Peter smiled, happy to know why the fish was there.
After they said their goodbyes, Peter kept walking down the river, looking for more wild minerals.
<|endoftext|>
Once upon a time, in a small house, there was a deep hole. The hole was so deep that no one could see the bottom. A family lived in the house, and they had a problem. The deep hole was under their roof, and they needed support to fix it.
One day, a big, strong animal came to the house. It saw the deep hole under the roof and wanted to help. The animal used its strong legs to support the roof. It stood there all day and all night, making sure the roof did not fall into the deep hole.
The family was very happy that the animal came to support their roof. They thanked the animal and gave it food to eat. From that day on, the deep hole was not a problem anymore. The family and the animal lived happily together, and the roof stayed strong with the support of their new friend.
<|endoftext|>
Once upon a time, there was a curious little boy named Tim. He liked to mind his mom and dad. One day, Tim saw a hunter in the woods. The hunter had a big hat and a long stick. Tim thought the hunter was looking for something.
Tim followed the hunter, but stayed far away. He saw the hunter put a small box under a tree. Tim was very curious. He wanted to know what was in the box. The hunter left and Tim went to look at the box.
When Tim got to the box, he saw that it was filled with yummy treats! He knew that the hunter had left it for someone special. Tim decided to mind his mom and dad, and not take the treats. He went back home and told them about the hunter and the box. They were proud of Tim for minding them and being a good boy.
<|endoftext|>

Once upon a time there was a scared turtle named Turtle. He was scared because he didn't recognize the work he had to do. He was scared of the things that he would need to do, so he stayed in his shell.
One day, he decided to explore. He slowly crept out of his shell and set out to explore the world. He met many friends who were kind and showed him how to do the work. Turtle was happy to learn how to do the things he was scared to do.
Soon enough, Turtle learned how to do the work and starting enjoying it! He loved learning how to do the work and he recognized the work he did. With his friends, he would often talk about the work he did. Turtle was no longer scared and was now feeling confident.
The End.
<|endoftext|>


Sara and Ben are friends. They like to play outside in the sun. But one day, the sky is not clear. It is dark and gray. Sara and Ben hear a loud sound. It is thunder.
"Let's go inside," Sara says. "The rain will come soon."
"No, I want to play more," Ben says. He does not like to stop playing.
Sara sees a big flash. It is lightning. She is scared. She runs to the door. "Ben, come on! The lightning is dangerous. It can hurt you."
Ben does not listen. He stays outside. He kicks a ball. He thinks he is brave.
But then, he feels a drop on his head. It is rain. It is cold and wet. He does not like it. He drops the ball. He runs to the door. But the door is closed.
"Help! Sara, open the door!" Ben shouts. He knocks on the door. He is wet and sad.
Sara hears Ben. She opens the door. She lets him in. She is angry. "Why did you not listen to me? I tried to prevent you from getting wet. You are silly."
Ben is sorry. He says, "I am sorry, Sara. You are right. I should have listened to you. You are smart. Can we still be friends?"
Sara smiles. She says, "Yes, we can still be friends. But next time, when the sky is not clear, we should go inside. OK?"
Ben nods. He says, "OK. I will listen to you next time. And maybe, next week, the sun will come back. And we can play outside again."
Sara and Ben hug. They are friends. They go to the living room. They play with toys. They wait for the rain to stop.
<|endoftext|>
Once upon a time, there was a little girl named Mia. She loved to study her big picture book. One day, while she was studying, she saw a picture of a broccoli. She had never seen a broccoli before, and she wanted to try it.
Mia went to her mom and said, "Mom, I saw a broccoli in my book. Can we try it?" Her mom smiled and said, "Yes, Mia. We can try it for dinner tonight." Mia was very happy and could not wait for dinner.
At dinner, Mia's friend, Lily, came over to eat with them. When they saw the broccoli, Lily felt envious. She wanted to try the broccoli too. Mia shared her broccoli with Lily, and they both loved it. From that day on, Mia and Lily always wanted to eat broccoli together.
<|endoftext|>
Once upon a time, in a small village, there was a boy named Tom. Tom had a toy spear. He loved to play with his spear all day long. One day, Tom felt uncomfortable. His tummy hurt a lot. He wondered what was wrong.
Tom went to his mom and asked her a question. "Mom, why does my tummy hurt?" His mom thought for a moment and said, "Maybe you ate too much candy, Tom." Tom knew he ate a lot of candy, so he thought his mom was right.
Tom's mom told him to rest and not play with his spear for a while. Tom listened to his mom and took a nap. When he woke up, he felt much better. Tom learned that eating too much candy can make him feel uncomfortable. He decided to eat less candy and play with his spear more.
<|endoftext|>

Steve wanted to surprise his friend Marie, so he went out and bought her a nice folder. When he saw her, Steve happily offered the folder to her. But when Marie opened it, she was disappointed. The folder was dull and plain. She was expecting something brighter and better.
Steve was disappointed that Marie was not happy with his offering. He did not know how to make it up to her. He thought and thought until he came up with an idea. He offered to decorate the folder with colourful stickers to make it look nice.
Marie was happy with this idea and agreed. They both worked together and put on lots of stickers until the folder was transformed! It looked much more exciting and colourful. Marie was happy with Steve's offering now, and thanked him for his thoughtfulness. They had finally managed to overcome the conflict and make Marie smile.
<|endoftext|>
'''
    tokens  = tokenizer.encode(my_text)
    print(tokens)
    text = tokenizer.decode(tokens)
    #print(text)
    # vocab = tokenizer.get_vocab()
    # merges = tokenizer.get_merges()
    # print(f'Vocab \n {"="*100} \n {vocab} \n {"="*100}')
    # print(f'Merges \n {"="*100} \n {merges} \n {"="*100}')

    # # Save vocab to pickle file
    # with open('data/tokenizers/ts_valid_vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    # print("Vocab saved to data/tokenizers/train_vocab.pkl")

    # # Save merges to pickle file
    # with open('data/tokenizers/ts_valid_merges.pkl', 'wb') as f:
    #     pickle.dump(merges, f)
    # print("Merges saved to data/tokenizers/train_merges.pkl")

    # encoded_text = tokenizer.encode("hello, world! <|endoftext|>")
    # print(f"Encoded text: {encoded_text}")
    # print(f"Decoded text: {tokenizer.decode(encoded_text)}")

    # with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f :
    #     data = f.read().decode("utf-8")
    #     r = find_chunk_boundaries(f, 12, "<|endoftext|>".encode('utf-8'))
    #     print(r)

        # r = pretokenize(data[:1875], ["<|endoftext|>"]) 
        # print(r)