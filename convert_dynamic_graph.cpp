#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct Event {
    int u = 0;
    int v = 0;
    int delta = 0;  // +1 for insertion, -1 for deletion
    long long ts = 0;
    std::size_t seq = 0;
};

static std::uint64_t directed_key(int u, int v) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
           static_cast<std::uint32_t>(v);
}

static std::uint64_t undirected_key(int u, int v) {
    if (u > v) std::swap(u, v);
    return directed_key(u, v);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dynamic_graph> <output_dynamic_graph>\n";
        return 1;
    }

    std::ifstream fin(argv[1]);
    if (!fin.is_open()) {
        std::cerr << "Cannot open input file: " << argv[1] << "\n";
        return 1;
    }

    int vertexCount = 0;
    long long eventCount = 0;
    if (!(fin >> vertexCount >> eventCount)) {
        std::cerr << "Invalid input header.\n";
        return 1;
    }

    std::vector<Event> events;
    events.reserve(static_cast<std::size_t>(std::max(0LL, eventCount)));

    int u = 0, v = 0;
    std::string op;
    long long ts = 0;
    std::size_t seq = 0;
    while (fin >> u >> v >> op >> ts) {
        Event e;
        e.u = u;
        e.v = v;
        e.ts = ts;
        e.seq = seq++;
        if (op == "+1") {
            e.delta = 1;
        } else if (op == "-1") {
            e.delta = -1;
        } else {
            std::cerr << "Invalid operation: " << op << "\n";
            return 1;
        }
        events.push_back(e);
    }

    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        if (a.ts != b.ts) return a.ts < b.ts;
        if (a.delta != b.delta) return a.delta > b.delta;  // +1 before -1
        return a.seq < b.seq;
    });

    std::unordered_set<std::uint64_t> activeDirected;
    std::unordered_set<std::uint64_t> activeUndirected;
    std::vector<Event> outputEvents;
    outputEvents.reserve(events.size());

    for (const Event& e : events) {
        std::uint64_t dirKey = directed_key(e.u, e.v);
        std::uint64_t undirKey = undirected_key(e.u, e.v);

        if (e.delta == 1) {
            bool hadUndirectedEdge = activeUndirected.find(undirKey) != activeUndirected.end();
            activeDirected.insert(dirKey);
            if (!hadUndirectedEdge) {
                activeUndirected.insert(undirKey);
                outputEvents.push_back(e);
            }
        } else {
            activeDirected.erase(dirKey);
            bool forwardExists = activeDirected.find(directed_key(e.u, e.v)) != activeDirected.end();
            bool backwardExists = activeDirected.find(directed_key(e.v, e.u)) != activeDirected.end();
            if (!forwardExists && !backwardExists && activeUndirected.erase(undirKey) > 0) {
                outputEvents.push_back(e);
            }
        }
    }

    std::ofstream fout(argv[2]);
    if (!fout.is_open()) {
        std::cerr << "Cannot open output file: " << argv[2] << "\n";
        return 1;
    }

    fout << vertexCount << ' ' << outputEvents.size() << '\n';
    for (const Event& e : outputEvents) {
        fout << e.u << ' ' << e.v << ' ' << (e.delta == 1 ? "+1" : "-1") << ' ' << e.ts << '\n';
    }

    return 0;
}
