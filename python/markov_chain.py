import random

class MarkovChain:
  def __init__(self, text, delimeter=" "):
    self._dict = {}
    tokens = None
    if delimeter is not None:
      tokens = text.split(delimeter)
    else:
      tokens = list(text)

    # Populate the dict
    for token in tokens:
      self._dict[token] = []

    for i in range(len(tokens) - 1):
      self._dict[tokens[i]].append(tokens[i + 1])

  # Given a token, return a token that could come after it
  def get_next_token(self, prev_token):
    try:
      if prev_token not in self._dict.keys():
        # Return a random token
        keys = list(self._dict.keys())
        rand_key = random.choice(keys)
        return random.choice(self._dict[rand_key])
      else:
        return random.choice(self._dict[prev_token])
    except IndexError:
      return "\n"

text = input("Input some text to test the algorithm: ")
markov_chain = MarkovChain(text)
token = "The"
for _ in range(200):
  print(token, end=" ")
  token = markov_chain.get_next_token(token)