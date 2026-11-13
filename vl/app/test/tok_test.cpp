// Prints the token ids of each input line (\n escapes allowed as "\\n"), one line of ids per input line.
#include <cstdio>
#include <iostream>
#include "../tokenizer.h"
int main(int argc, char** argv) {
    Tokenizer t(argv[1]);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string s; for (size_t i = 0; i < line.size(); i++) { if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') { s += '\n'; i++; } else s += line[i]; }
        auto ids = t.encode(s);
        for (size_t i = 0; i < ids.size(); i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\t%s\n", t.decode(ids, true) == s ? "RT" : "NORT");
    }
}
