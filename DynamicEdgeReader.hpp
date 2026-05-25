#pragma once

#include <fstream>
#include <string>

struct DynamicEdgeEvent {
    int u = -1;
    int v = -1;
    int delta = 0;  // +1 for insertion, -1 for deletion
    long long ts = 0;
};

class DynamicEdgeReader {
    int offset;

   public:
    explicit DynamicEdgeReader(const std::string& filePath, int offset_ = 0) {
        fin_.open(filePath);
        offset = offset_;
        if (!fin_.is_open()) return;
        if (!(fin_ >> vertexCount_ >> updateCount_)) {
            fin_.close();
        }
    }

    ~DynamicEdgeReader() {
        if (fin_.is_open()) fin_.close();
    }

    bool IsOpen() const { return fin_.is_open(); }
    int VertexCount() const { return vertexCount_; }
    long long UpdateCount() const { return updateCount_; }

    bool NextEdge(DynamicEdgeEvent& e) {
        std::string op;
        bool ok = false;
        while (1) {
            if (!(fin_ >> e.u >> e.v >> op >> e.ts)) {
                return false;
            }
            if (e.u != e.v) break;
        }
        e.u -= offset;
        e.v -= offset;
        if (op == "+1")
            e.delta = 1;
        else if (op == "-1")
            e.delta = -1;
        else
            return false;
        return true;
    }

   private:
    std::ifstream fin_;
    int vertexCount_ = 0;
    long long updateCount_ = 0;
};
