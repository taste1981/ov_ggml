# Generated outputs on Intel Core Ultra 9 285K (Arrow Lake)

This report records saved output excerpts for the nine Q4_K_M models in [Performance notes](PERFORMANCE_NOTES.md), plus selected follow-up runs on the same iGPU.

- Each cell shows up to the first 160 characters saved in the batch summary. Prompt labels are nominal; the source summaries list actual token counts.
- Prompt texts: [5](./prompts/prompt-5.txt), [250](./prompts/prompt-250.txt), [1000](./prompts/prompt-1000.txt), and [2000](./prompts/prompt-2000.txt).
- **Failed** means the path did not complete. **Not tested** means it was not run in that batch. A completed run or a readable excerpt does not establish output correctness or token-level agreement.
- Batch IDs identify `summary-<batch>.md` in the corresponding model folder (or its `perf-compare` subfolder) under `llama-work-note/results/compare-gpu-4paths`.

## Contents

- [Main batches](#main-batches)
  - [Llama-3.2-1B-Instruct Q4_K_M](#llama-32-1b-instruct-q4_k_m)
  - [Llama-3.1-8B-Instruct Q4_K_M](#llama-31-8b-instruct-q4_k_m)
  - [Qwen3-4B Q4_K_M](#qwen3-4b-q4_k_m)
  - [Qwen3.5-4B Q4_K_M](#qwen35-4b-q4_k_m)
  - [Nanbeige4.2-3B Q4_K_M](#nanbeige42-3b-q4_k_m)
  - [Phi-4-mini-instruct Q4_K_M](#phi-4-mini-instruct-q4_k_m)
  - [Gemma-4-E2B-IT Q4_K_M](#gemma-4-e2b-it-q4_k_m)
  - [Gemma-4-E4B-IT Q4_K_M](#gemma-4-e4b-it-q4_k_m)
  - [Gemma-4-12B-IT Q4_K_M](#gemma-4-12b-it-q4_k_m)
- [Selected follow-up runs](#selected-follow-up-runs)
  - [Llama-3.2-1B-Instruct: GGUF input modes](#llama-32-1b-instruct-gguf-input-modes)
  - [Qwen3.5-4B: one-token GGML graph](#qwen35-4b-one-token-ggml-graph)
  - [Laguna-XS-2.1 Q4_K_M](#laguna-xs-21-q4_k_m)

## Main batches

### Llama-3.2-1B-Instruct Q4_K_M

Batch: `20261120-201355`. OVFE produced only exclamation marks at prompt 5 and garbled text at the three longer prompts. At prompt 250, llama.cpp + OV text varied across measured runs.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris. The Eiffel Tower is a famous landmark in Paris. The Eiffel Tower is a symbol of France and is one of the most visited tourist | Paris. Paris is a city located in the north of France, and it is known for its rich history, art, and culture. The city is home to | Paris. The Eiffel Tower is a famous landmark in Paris. The Eiffel Tower is a symbol of France and is one of the most visited tourist | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! | Not tested |
| 250 | The history of the printing press began with the experimentation of craftsmen in China, Korea, and other regions, who developed movable type and printing techni | The history of the printing press began with the experimentation of craftsmen in China, Korea, and other parts of Asia, who developed movable type and printing | The history of the printing press began with the experimentation of craftsmen in China, Korea, and other regions, who developed movable type and printing techni | The history them &#91; through within within 640 s &#91; G점  D still79 over Gsequencesior-inablo counsel become DEss--double Bon | Not tested |
| 1000 | The question of where a ship was on the open sea had no precise answer in most of human history. Sailors used a combination of observation, memory, | The question of where a ship was on the open sea had no precise answer in most of human history. Sailors developed various methods to estimate their position, | The history of navigation is not well-documented, and the question of where a ship was on the open sea had no precise answer. Sailors used a combination | The-block Tuneadv- viewthe hourстанdtarness-t. co this-lengthipoaganazio- Vers VersNET?- window <br> and some some known | Not tested |
| 2000 | The question of where a ship was on the open sea had no precise answer for most of human history. Sailors used various methods, including reading the sea | The question of where a ship was on the open sea had no precise answer in most of human history. Sailors developed various methods to estimate their position, | The question of where a ship was on the open sea had no precise answer for most of human history, and navigators relied on a combination of observation, | Theivating such-حةeduc traveled depafortanguttonulligerstrict ganlettevieretroangelatearet thuselts earliestenn standarduetgradic wageserationuration | Not tested |

### Llama-3.1-8B-Instruct Q4_K_M

Batch: `20261123-170740`. OVFE samples were garbled or repetitive at all four prompt lengths. At prompt 250, llama.cpp + OV text varied across measured runs.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris, which is located in the north-central part of the country. Paris is a global center for art, fashion, cuisine, and culture, and is one | Paris, which is located in the north-central part of the country. Paris is a global center for art, fashion, cuisine, and culture, and is one | Paris, which is located in the north-central part of the country. Paris is a global center for art, fashion, cuisine, and culture, and is one | assenassenassenoirCPstrapTeskTenkenkTLTLTLTLTLTLTLTLTLTLTLTTLTTL | a city of grandeur and beauty, with a rich history dating back to the 3rd century BC. Paris is a city that has been shaped by the |
| 250 | The history of the printing press began with Chinese craftsmen carving wooden blocks, then moved to movable type in clay and bronze, and finally culminated in G | The history of the printing press began in China with wooden block printing, was improved by movable type in Korea and China, and was finally perfected by Guten | The history of the printing press began with Chinese craftsmen carving wooden blocks, then moved to movable type in clay and bronze, and finally culminated in G | The history of the printing press began with Chinese craftsmen carving text into wooden blocks, then moved was was the, of the, Bi Sheng,, and | The history of the printing press began in China with wooden block printing, was improved by movable type made of baked clay and bronze, and was finally perfect |
| 1000 | The history of navigation is a story of how sailors learned to fix their position on the open sea. Early sailors relied on observation, memory, and educated gue | The history of navigation is a story of how sailors learned to fix their position far from land. Early navigators used observations of the sea, stars, and | The history of navigation is a story of how sailors learned to fix their position on the open sea. Early navigators relied on observation, memory, and educated | The text of, a (." only and  Version on &#91;the&#93; by to, and: (  everyone is of) be: in until the | The history of navigation is a story about the sky, clocks, and the accumulation of knowledge. Early sailors used observation, memory, and educated guesswork to |
| 2000 | The development of navigation techniques over time has allowed sailors to determine their position with increasing accuracy. Early navigators relied on observat | The history of navigation is a story of how sailors and navigators learned to fix their position far from land, using a combination of observation, memory, and | The history of navigation is a story about the slow accumulation of knowledge by people who rarely wrote down what they knew. From ancient Polynesian voyagers t | The, the and to ( the of (  and you physics,, and a,"<br><br>, ands;s the,, in,,  to | The history of navigation is a story about the sky, clocks, and the accumulation of knowledge by people who rarely wrote down what they knew. Early sailors used |

### Qwen3-4B Q4_K_M

Batch: `20261120-161106`. OVFE produced only exclamation marks at prompt 5 and repetitive or garbled text at longer prompts. At prompt 5, llama.cpp + OV repeated "The capital of Paris is Paris."; at prompt 1000, its text varied across measured runs.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital | Paris. The capital of Paris is Paris. The capital of Paris is Paris. The capital of Paris is Paris. The capital of Paris is Paris. The capital | Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! | Not tested |
| 250 | The history of the printing press began with the invention of movable type in China, followed by its development in Korea and the refinement by Gutenberg in Eur | The history of the printing press began in China with wooden block printing, evolved through Bi Sheng's movable type, spread to Korea with bronze type, and | The history of the printing press began with the invention of movable type in China, followed by its development in Korea and the refinement by Gutenberg in Eur | The history of the printing press began in China with wooden block printing, then moved to Korea with bronze movable-type and the to the the-2–1 | Not tested |
| 1000 | (1) The text discusses the historical development of navigation techniques, (1) the challenges of determining longitude, and (2) the eventual solution to the lo | (1) The text discusses the historical development of methods for determining a ship's position at sea, starting from ancient techniques to modern navigation. (2 | (1) The history of navigation before the compass; (2) The development of the compass and its use in navigation; (3) The problem of longitude | (1/2 of the 2... (a) and ( (str h/ 2" 2 and " / "s-1-6 | Not tested |
| 2000 | (1) The text discusses the historical development of methods for determining a ship's position at sea, (2) The key challenge was longitude, which required accur | (1) The text discusses the historical development of methods for determining a ship's position at sea, from ancient techniques to modern satellite navigation. ( | (1) The text discusses the historical development of methods for determining a ship's position at sea, (2) The key challenge was longitude, which required accur | (theborne/w<br><br><br><br>（b/s<br><br> and a - FarJet diary Betty交四<br><br>g<br>K andz- '"". herself for | Not tested |

### Qwen3.5-4B Q4_K_M

Batch: `20261120-153933`. GenAI + Vulkan rejected the `b1024` prefill graph for this recurrent model. OVFE samples were garbled at all four prompt lengths. At prompt 5, llama.cpp + OV text varied across measured runs.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris.<br>A. True<br>B. False<br>Answer:<br>A<br><br>Which of the following is NOT a characteristic of the Mediterranean climate?<br>A. | Paris.<br>A. True<br>B. False<br><br>&#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42; The user is presenting a statement | **Failed** (`exit2`) | jun为保证 séquさ对中所有的时间来Sharp大路 esprit年以上的上值守 Notíciasaiposa    <br>    <br>    <br>    <br> ｢初宜居刻 truly &#38; замести Detalles该局应运echa "oughсловно | Not tested |
| 250 | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text describing the history of the printing press, | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text describing the history of the printing press, | **Failed** (`exit2`) | or as the more's<br><br><br> just<br> is- The the - "a was3 no by is with the<br> noS<br><br>. | Not tested |
| 1000 | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text about the history of navigation, specifically focusing | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Source text: A passage about the history of navigation, specifically | **Failed** (`exit2`) | &#42;&#42;The just<br><br> new HadsA haveon- somethingASun another a other other a new A misseds another as still in ae | Not tested |
| 2000 | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text about the history of navigation and the development | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text about the history of navigation and the development | **Failed** (`exit2`) | is<br><br>.<br>某一坚持在知<br>三据<br> are en一位一位准在一位一一位某某仅的…… | Not tested |

### Nanbeige4.2-3B Q4_K_M

Batch: `20261124-112312`. Both GenAI + OV modes rejected the `nanbeige` architecture and produced no text. Both llama.cpp paths and GenAI + Vulkan produced text at all four prompt lengths. At prompts 1000 and 2000, GenAI + Vulkan logged 985 and 1969 input tokens, versus 979 and 1957 for both llama.cpp paths; these excerpts do not establish matched-input or token-level agreement. See the [performance results](PERFORMANCE_NOTES.md#nanbeige42-3b-q4_k_m).

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris.<br>&#60;/think&#62;<br><br>The capital of France is &#42;&#42;Paris&#42;&#42;. 🇫🇷<br><br>If you have any other questions— | Paris.<br><br>Let me provide a comprehensive answer about Paris as the capital of France:<br><br>&#42;&#42;Paris&#42;&#42; is the capital city of France. Here are | Paris.<br>&#60;/think&#62;<br><br>The capital of France is &#42;&#42;Paris&#42;&#42;. 🇫🇷<br><br>If you have any other questions | **Failed** (`exit2`) | **Failed** (`exit2`) |
| 250 | Answer: The history of the printing press began with woodblock printing in China, followed by movable type innovations in China and Korea, and finally by Gutenb | Answer: The history of the printing press began with woodblock printing in China, followed by movable type innovations in China and Korea, and finally culminate | Answer: The history of the printing press began with woodblock printing in China, followed by movable type innovations in China and Korea, and finally by Gutenb | **Failed** (`exit2`) | **Failed** (`exit2`) |
| 1000 | 1. The text discusses how navigators determined their position on the open sea far from land.<br>2. Different methods were used by various cultures to | 1. Early sailors used natural observations and memorized knowledge to navigate the open sea, while later navigators developed instruments and techniques to dete | 1. The text discusses how navigators fixed their position on the open sea far from land.<br>2. Different methods were used by various cultures to | **Failed** (`exit2`) | **Failed** (`exit2`) |
| 2000 | 1. The text discusses how navigators learned to determine their position on the open sea far from land.<br>2. The text discusses how navigators | 1. Early navigators relied on natural observations (stars, ocean swells, birds, clouds) and developed methods to determine latitude using the Pole Star | 1. Early navigators used various methods to determine their position on the open sea, including reading the sea and stars, Arab/Persian latitude measurement | **Failed** (`exit2`) | **Failed** (`exit2`) |

### Phi-4-mini-instruct Q4_K_M

Batch: `20261124-142946`. The two Vulkan paths have matching saved excerpts at prompts 5 and 250, but different excerpts at 1000 and 2000. At prompts 250, 1000, and 2000, GenAI logged 235, 991, and 1979 input tokens, versus 234, 967, and 1932 for llama.cpp. OVFE produced garbled text for all four prompts; the legacy reader failed during GGUF loading. This batch used no chat template and ignored EOS, so it did not verify stopping at `<|end|>`. See the [performance results](PERFORMANCE_NOTES.md#phi-4-mini-instruct-q4_k_m).

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris. Paris is the capital of France. Paris is located in France. Paris is the capital of France. Paris is located in France. Paris is the capital | Paris. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The Eiffel Tower, constructed in 188 | Paris. Paris is the capital of France. Paris is located in France. Paris is the capital of France. Paris is located in France. Paris is the capital | ?<br> PriorThough joint foundation attDebConvers RelEnc promoted supports peer stem periodically EvansDirect expansion paths prediction successfully healthy joins | **Failed** (`exit2`) |
| 250 | Gutenberg's invention of the printing press combined existing ideas into a practical system that revolutionized the production of books, spreading knowledge and | Gutenberg's printing press combined metal type, oil-based ink, and a screw press to produce pages faster than scribes could copy by hand, spreading printing sho | Gutenberg's invention of the printing press combined existing ideas into a practical system that revolutionized the production of books, spreading knowledge and | The, mounting chemical electrical Mark electrical Tom substances Wh restrictions pie chemical Henry electrical wording Ex Wh Pe Wh Wh , oneationeting-se prin | **Failed** (`exit2`) |
| 1000 | Navigators used a combination of celestial navigation, dead reckoning, and timekeeping to determine their position at sea. Polynesian voyagers relied on memoriz | Navigators used a combination of celestial navigation, dead reckoning, and timekeeping to determine their position at sea. Polynesian voyagers relied on memoriz | The first navigators used the stars, the sea and the flight of birds to find their way across the oceans. Later, they used the magnetic compass | Thea first both once to outside  in both Round  La  a or whichA  -- "; Secret:, to to- unexpectedly together onlyCombine | **Failed** (`exit2`) |
| 2000 | Navigators learned to fix their position far from land by observing the sky, using instruments like the magnetic compass and astrolabe, and later by measuring t | Navigators have always struggled to determine their position at sea. Over time, they developed a variety of tools and methods to measure latitude and longitude, | Navigators learned to fix their position far from land by observing the sky, using clocks, and accumulating knowledge over centuries. The magnetic compass and d | Thestarting indefinitely Type experts positively Field; fiveprocess Excfr Gra Gra BrickConbus protonopen successfully simultaneously enjoyed respectively-cut Ex | **Failed** (`exit2`) |

### Gemma-4-E2B-IT Q4_K_M

Batch: `20261120-170550`. llama.cpp + OV stopped with an access violation (`0xC0000005`); OVFE failed with `CL_BUILD_PROGRAM_FAILURE`.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris.<br><br>&#42;&#42;Paris is the capital of France.&#42;&#42;<br><br>---<br>&#42;&#42;Question:&#42;&#42; What is the capital of France?<br><br>&#42;&#42;Answer:&#42;&#42; Paris. (or | **Failed** (`exit-1073741819`) | Paris. | **Failed** (`exit2`) | Not tested |
| 250 | &#42;&#42;The history of the printing press evolved from woodblock printing in China, through early movable type experiments in China and Korea, culminating in Gutenber | **Failed** (`exit-1073741819`) | The history of printing evolved from woodblock printing in China, through early movable type experiments in China and Korea, culminating in Gutenberg's practica | **Failed** (`exit2`) | Not tested |
| 1000 | 1. Navigators developed sophisticated methods based on celestial observation, memory, and empirical knowledge to determine their position far from land.<br>2. Earl | **Failed** (`exit-1073741819`) | 1. Early navigators relied on memorized celestial patterns and wave patterns for long-distance travel, while later methods introduced instruments like the magne | **Failed** (`exit2`) | Not tested |
| 2000 | 1. Early navigation relied on celestial observation, memory, and the sea's features, with methods evolving from memorizing star patterns for Polynesian voyagers | **Failed** (`exit-1073741819`) | 1. Early navigation relied on memorizing celestial patterns and wave patterns, while later methods introduced instruments like the magnetic compass and dead rec | **Failed** (`exit2`) | Not tested |

### Gemma-4-E4B-IT Q4_K_M

Batch: `20261120-172350`. llama.cpp + OV stopped with an access violation (`0xC0000005`); OVFE failed with `CL_BUILD_PROGRAM_FAILURE`.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | Paris.<br><br>&#42;&#42;Question:&#42;&#42; What is the capital of France?<br><br>&#42;&#42;Answer:&#42;&#42; Paris. (Or, "The capital of France is Paris.")<br><br>&#42;&#42;&#42; | **Failed** (`exit-1073741819`) | Paris.<br><br>&#42;&#42;Question:&#42;&#42; What is the capital of France?<br>&#42;&#42;Answer:&#42;&#42; Paris. | **Failed** (`exit2`) | Not tested |
| 250 | &#42;&#42;The history of the printing press evolved from early woodblock printing in China, through the development of movable type in clay and bronze in China and Kore | **Failed** (`exit-1073741819`) | &#42;&#42;The history of the printing press evolved from early woodblock printing in China, through the development of movable type in clay and bronze in China and Kore | **Failed** (`exit2`) | Not tested |
| 1000 | 1. Early navigation relied on diverse, non-instrumental methods, such as Polynesian reliance on celestial patterns and ocean swells, or Arab sailors using the P | **Failed** (`exit-1073741819`) | 1. Early navigation relied on diverse methods, ranging from Polynesian reliance on natural signs like stars and swells to Mediterranean sailors using the Pole S | **Failed** (`exit2`) | Not tested |
| 2000 | 1. &#42;&#42;Early navigation relied on diverse, localized methods—from Polynesian celestial memory and Mediterranean latitude sailing to the magnetic compass and dead | **Failed** (`exit-1073741819`) | 1. Before modern instruments, navigation relied on diverse methods—from Polynesian observation of the sea and stars to Mediterranean latitude sailing—but these | **Failed** (`exit2`) | Not tested |

### Gemma-4-12B-IT Q4_K_M

Batch: `20261120-174348`. At prompt 2000, the llama.cpp + OV sample repeated channel markers and digits, and its text varied across measured runs.

| Prompt | llama.cpp + Vulkan | llama.cpp + OV | GenAI + Vulkan | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- | --- | --- | --- |
| 5 | a&#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;The capital of France is &#42;&#42;Paris&#42;&#42;. (The letter "d" would be the correct answer if this were a multiple-choice question | a&#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;The capital of France is &#42;&#42;Paris&#42;&#42;. (Note: The word "France" is spelled with an "r" and "Paris" | athought<br>The capital of France is &#42;&#42;Paris&#42;&#42;.<br><br>(Note: In your sentence, the correct word is "capital" rather than " | athought<br>The capital of France is &#42;&#42;Paris&#42;&#42;.<br>thought<br>thought<br>The capital of France is &#42;&#42;Paris&#42;&#42;. | Not tested |
| 250 | &#60;&#124;channel&#62;thought<br>&#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;The history of the printing press evolved from Chinese woodblock and clay movable type and Korean bronze casting t | &#60;&#124;channel&#62;thought<br>&#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;The history of the printing press began with Chinese and Korean innovations in woodblock and movable type, which w | thought<br>thought<br>The history of the printing press evolved from ancient Chinese woodblock and clay movable type and Korean bronze casting to Johannes Gutenberg’ | thought<br>thought<br>The history of the printing press began with Chinese and Korean innovations in woodblock and clay movable type, which were eventually synthesize | Not tested |
| 1000 | &#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;Historically, navigators relied on a combination of natural observation, such as memorizing star patterns and reading ocean swells, | &#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;Here is a three-sentence summary of the text:<br><br>Early navigators relied on nature and memory, using celestial bodies, ocean swells | thoughtHistorically, navigators relied on a combination of natural observation, such as memorizing star patterns and reading ocean swells, to determine their po | thought<br>Here is a three-sentence summary of the text:<br><br>Early navigators relied on natural observations, such as celestial patterns, ocean swells | Not tested |
| 2000 | &#60;&#124;channel&#62;thought<br>&#60;channel&#124;&#62;Historically, navigators relied on a combination of natural observation—such as stars, waves, and bird flights—and "dead reckoning" | &#60;&#124;channel&#62;&#95;&#60;&#124;channel&#62;&#95;&#60;&#124;channel&#62;&#95;1.<br>&#60;&#124;channel&#62;thing&#60;&#124;channel&#62;&#60;&#124;channel&#62;0.<br>&#60;&#124;channel&#62;thing$&#92;color{1.<br>&#60;&#124;channel&#62;thing$&#92;color{11 | thought<br>thought<br>Historically, navigators relied on a combination of natural observation—such as stars, swells, and bird flights—and " | thought<br>Here is a three-sentence summary of the text:<br><br>Early navigation relied on the observation of natural signs, such as stars, waves | Not tested |

## Selected follow-up runs

### Llama-3.2-1B-Instruct: GGUF input modes

Batches: `20261123-164415` (OVFE) and `20261123-164244` (reader). Both used the out-of-order queue with the same short prompt. OVFE logged 32 output IDs of `0`; the reader sample began with `Paris.`. Other paths were not tested in these batches.

| Prompt | GenAI + OV (ovfe) | GenAI + OV (reader) |
| ---: | --- | --- |
| 5 | !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! | Paris. Paris is a city located in the north of France. It is known for its rich history, art, fashion, and cuisine. Paris is also home |

### Qwen3.5-4B: one-token GGML graph

Batches: `20261123-162258` (prompt 5) and `20261123-163251` (prompt 250). The two Vulkan paths completed with `b1` for GenAI and `-ub 1` for llama.cpp. At prompt 250, the logs reported 236 input tokens for llama.cpp and 237 for GenAI; these excerpts are not a matched-input token comparison. Other paths were not tested.

| Prompt | llama.cpp + Vulkan | GenAI + Vulkan |
| ---: | --- | --- |
| 5 | Paris.<br>A. True<br>B. False<br>Answer:<br>A<br><br>Which of the following is NOT a characteristic of the Mediterranean climate?<br>A. | Paris.<br>A. True<br>B. False<br>Answer:<br>A<br><br>Which of the following is NOT a characteristic of the Mediterranean climate?<br>A. |
| 250 | &#60;think&#62;<br>Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text describing the history of the printing press, | Thinking Process:<br><br>1.  &#42;&#42;Analyze the Request:&#42;&#42;<br>    &#42;   Input: A text describing the history of the printing press, |

### Laguna-XS-2.1 Q4_K_M

Batch: `20261123-143432`. Only GenAI + Vulkan was tested. All four runs completed, but each raw log shows 32 output IDs of `0` and no decoded text.

| Prompt | GenAI + Vulkan |
| ---: | --- |
| 5 | **Empty output** (32 IDs of `0`) |
| 250 | **Empty output** (32 IDs of `0`) |
| 1000 | **Empty output** (32 IDs of `0`) |
| 2000 | **Empty output** (32 IDs of `0`) |
