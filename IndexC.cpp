#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ArrayOnHeap.hpp"

// #region agent log
#include <cstdio>
#define DBG_LOG(hyp, loc, msg, data_fmt, ...)                                                                        \
    do {                                                                                                             \
        FILE* _f = fopen("/home/srChen/Desktop/M2HL/.cursor/debug-22c2a6.log", "a");                                 \
        if (_f) {                                                                                                    \
            fprintf(                                                                                                 \
                _f,                                                                                                  \
                "{\"sessionId\":\"22c2a6\",\"hypothesisId\":\"%s\",\"location\":\"%s\",\"message\":\"%s\"," data_fmt \
                "}\n",                                                                                               \
                hyp, loc, msg, ##__VA_ARGS__);                                                                       \
            fclose(_f);                                                                                              \
        }                                                                                                            \
    } while (0)
// #endregion

#define MAXINT ((unsigned)4294967295)

// #region agent log
static inline void DBG_LOG_SESSION(const char* runId, const char* hyp, const char* loc, const char* msg,
                                   const std::string& dataJson) {
    FILE* f = fopen("/home/srChen/Desktop/M2HL/.cursor/debug-065374.log", "a");
    if (!f) return;
    long long ts =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    fprintf(f,
            "{\"sessionId\":\"065374\",\"runId\":\"%s\",\"hypothesisId\":\"%s\",\"location\":\"%s\","
            "\"message\":\"%s\",\"data\":%s,\"timestamp\":%lld}\n",
            runId, hyp, loc, msg, dataJson.c_str(), ts);
    fclose(f);
}
// #endregion

using namespace std;

#define MAXD 120
unsigned MAXDIS, MAXMOV, MASK;
vector<vector<unsigned>> con, conD;
ArrayOnHeap<vector<unsigned>> label, delete_labels, clab, Pla;
ArrayOnHeap<vector<int>> pos, cpos;
vector<pair<int, int>> v2degree;
vector<int> v2p, p2v, flg, vaff;  // 删了label的顶点

vector<pair<int, int>> DyEdges;

int totalV = 0, Dcnt = 1000, dv = MAXD, minTgt;

string StripExt(const string& filename) {
    size_t dot = filename.rfind('.');
    if (dot == string::npos) return filename;
    return filename.substr(0, dot);
}

// === input parameters === //
string Graphpath, Indexpath;
int program_choice, threads, dynamic_edge_num, query_task_num;
float active_ratio;

void Parameter() {
    MAXDIS = 2;
    MAXMOV = 1;
    while (MAXINT / (totalV * 2) >= MAXDIS) {
        MAXDIS *= 2;
        ++MAXMOV;
    }
    MASK = MAXDIS - 1;
}

void GraphInitial(string filename) {
    string s;
    const char* filepath = filename.c_str();
    ifstream infile;

    infile.open(filepath);
    if (!infile.is_open()) {
        cout << "Cannot find the original graph file!" << endl;
        exit(-1);
    }

    long xx = 0;

    while (getline(infile, s)) {
        ArrayOnHeap<char> strc(strlen(s.c_str()) + 1, ArrayOnHeap<char>::uninitialized);
        strcpy(strc.data(), s.c_str());
        char* s1 = strtok(strc.data(), " ");

        if (xx == 0) {
            totalV = atoi(s1);
            con.resize(totalV);
            conD.resize(totalV);
            Parameter();
        } else {
            while (s1) {
                int va = xx - 1, vb = atoi(s1);  //
                conD[va].push_back(vb);
                s1 = strtok(NULL, " ");
            }
        }

        xx += 1;
    }

    infile.close();

    for (int i = 0; i < totalV; ++i) v2degree.push_back(make_pair(conD[i].size(), -i));  //

    sort(v2degree.rbegin(), v2degree.rend());

    v2p.resize(totalV, -1);
    p2v.resize(totalV, -1);

    for (int i = 0; i < v2degree.size(); ++i) {
        v2p[-v2degree[i].second] = i;  // old 2 new
        p2v[i] = -v2degree[i].second;  // new 2 old
    }

    for (int i = 0; i < v2degree.size(); ++i) {
        int ovid = p2v[i];

        vector<unsigned>& local = conD[ovid];

        for (int p = 0; p < local.size(); ++p) {
            int jj = local[p];
            int j = v2p[jj];

            con[j].push_back(i);
        }

        vector<unsigned>().swap(local);
    }
}

void GraphReorder() {
    // #region agent log
    DBG_LOG_SESSION(
        "pre-fix", "H-entry", "GraphReorder:entry", "GraphReorder entry snapshot",
        std::string("{\"totalV\":") + std::to_string(totalV) + ",\"con_size\":" + std::to_string(con.size()) +
            ",\"label_size\":" + std::to_string(label.size()) + ",\"pos_size\":" + std::to_string(pos.size()) +
            ",\"v2p_size\":" + std::to_string(v2p.size()) + ",\"p2v_size\":" + std::to_string(p2v.size()) + "}");
    // #endregion

    v2degree.clear();
    v2degree.reserve(totalV);
    for (int i = 0; i < totalV; ++i) v2degree.push_back(make_pair((int)con[i].size(), -i));  // i is old id
    sort(v2degree.rbegin(), v2degree.rend());

    vector<int> old2new(totalV, -1), new2old(totalV, -1);
    for (int i = 0; i < totalV; ++i) {
        int oldVid = -v2degree[i].second;
        old2new[oldVid] = i;
        new2old[i] = oldVid;
    }

    vector<vector<unsigned>> conNew(totalV);
    for (int newVid = 0; newVid < totalV; ++newVid) {
        int oldVid = new2old[newVid];
        if (oldVid < 0 || oldVid >= totalV) {
            // #region agent log
            DBG_LOG_SESSION("pre-fix", "H-map", "GraphReorder:con-loop", "new2old out of range before con remap",
                            std::string("{\"newVid\":") + std::to_string(newVid) + ",\"oldVid\":" +
                                std::to_string(oldVid) + ",\"totalV\":" + std::to_string(totalV) + "}");
            // #endregion
            continue;
        }
        conNew[newVid].reserve(con[oldVid].size());
        for (int i = 0; i < con[oldVid].size(); ++i) {
            unsigned oldAdj = con[oldVid][i];
            conNew[newVid].push_back((unsigned)old2new[oldAdj]);
        }
        sort(conNew[newVid].begin(), conNew[newVid].end());
    }
    con.swap(conNew);

    if (!label.empty() && !pos.empty()) {
        ArrayOnHeap<vector<unsigned>> labelNew(totalV);
        ArrayOnHeap<vector<int>> posNew(totalV);

        for (int newVid = 0; newVid < totalV; ++newVid) {
            int oldVid = new2old[newVid];
            if (oldVid < 0 || oldVid >= (int)pos.size() || oldVid >= (int)label.size()) {
                // #region agent log
                DBG_LOG_SESSION("pre-fix", "H-pos-index", "GraphReorder:label-loop",
                                "oldVid invalid for pos/label access before assignment",
                                std::string("{\"newVid\":") + std::to_string(newVid) + ",\"oldVid\":" +
                                    std::to_string(oldVid) + ",\"pos_size\":" + std::to_string(pos.size()) +
                                    ",\"label_size\":" + std::to_string(label.size()) +
                                    ",\"totalV\":" + std::to_string(totalV) + "}");
                // #endregion
                continue;
            }
            posNew[newVid] = pos[oldVid];

            labelNew[newVid].reserve(label[oldVid].size());
            for (int i = 0; i < label[oldVid].size(); ++i) {
                unsigned oldHub = label[oldVid][i] >> MAXMOV;
                unsigned dis = label[oldVid][i] & MASK;
                if (oldHub >= (unsigned)totalV) {
                    // #region agent log
                    DBG_LOG_SESSION(
                        "pre-fix", "H-hub", "GraphReorder:label-loop", "oldHub out of range before old2new index",
                        std::string("{\"newVid\":") + std::to_string(newVid) + ",\"oldVid\":" + std::to_string(oldVid) +
                            ",\"label_idx\":" + std::to_string(i) + ",\"oldHub\":" + std::to_string(oldHub) +
                            ",\"totalV\":" + std::to_string(totalV) +
                            ",\"label_size\":" + std::to_string(label[oldVid].size()) + "}");
                    // #endregion
                    continue;
                }
                unsigned newHub = (unsigned)old2new[oldHub];
                labelNew[newVid].push_back((newHub << MAXMOV) | dis);
            }

            for (int d = 0; d < posNew[newVid].size(); ++d) {
                int l = d == 0 ? 0 : posNew[newVid][d - 1];
                int r = posNew[newVid][d];
                if (l < 0 || r < l || r > (int)labelNew[newVid].size()) {
                    // #region agent log
                    DBG_LOG_SESSION(
                        "pre-fix", "H-pos-range", "GraphReorder:sort-loop", "pos range invalid before segment sort",
                        std::string("{\"newVid\":") + std::to_string(newVid) + ",\"d\":" + std::to_string(d) +
                            ",\"l\":" + std::to_string(l) + ",\"r\":" + std::to_string(r) +
                            ",\"labelNew_size\":" + std::to_string(labelNew[newVid].size()) +
                            ",\"posNew_size\":" + std::to_string(posNew[newVid].size()) + "}");
                    // #endregion
                    continue;
                }
                sort(labelNew[newVid].begin() + l, labelNew[newVid].begin() + r);
            }
        }

        label = std::move(labelNew);
        pos = std::move(posNew);
    }

    // Keep the original semantics:
    // v2p maps original-id -> current-id, p2v maps current-id -> original-id.
    if ((int)v2p.size() == totalV && (int)p2v.size() == totalV) {
        vector<int> new_v2p(totalV, -1), new_p2v(totalV, -1);

        for (int orig = 0; orig < totalV; ++orig) {
            int oldId = v2p[orig];
            if (oldId >= 0) new_v2p[orig] = old2new[oldId];
        }

        for (int newVid = 0; newVid < totalV; ++newVid) {
            int oldVid = new2old[newVid];
            new_p2v[newVid] = p2v[oldVid];
        }

        v2p.swap(new_v2p);
        p2v.swap(new_p2v);
    } else {
        // Fallback: treat old ids as original ids.
        v2p = old2new;
        p2v = new2old;
    }
}

void GraphReorder_vp(vector<int> new_v2p, vector<int> new_p2v) {
    if (new_v2p.empty() && new_p2v.empty()) return;

    auto check_bijection = [&](const vector<int>& mapping, const char* name) {
        assert((int)mapping.size() == totalV && "mapping size != totalV");
        vector<bool> seen(totalV, false);
        for (int i = 0; i < totalV; ++i) {
            assert(mapping[i] >= 0 && mapping[i] < totalV && "mapping value out of range [0, totalV)");
            assert(!seen[mapping[i]] && "duplicate value in mapping, not a bijection");
            seen[mapping[i]] = true;
        }
    };

    if (!new_v2p.empty()) check_bijection(new_v2p, "new_v2p");
    if (!new_p2v.empty()) check_bijection(new_p2v, "new_p2v");

    if (new_v2p.empty()) {
        new_v2p.resize(totalV);
        for (int i = 0; i < totalV; ++i) new_v2p[new_p2v[i]] = i;
    } else if (new_p2v.empty()) {
        new_p2v.resize(totalV);
        for (int i = 0; i < totalV; ++i) new_p2v[new_v2p[i]] = i;
    }

    vector<int> old2new(totalV, -1), new2old(totalV, -1);
    for (int cur = 0; cur < totalV; ++cur) {
        int orig = p2v[cur];
        int newId = new_v2p[orig];
        old2new[cur] = newId;
        new2old[newId] = cur;
    }

    vector<vector<unsigned>> conNew(totalV);
    for (int newVid = 0; newVid < totalV; ++newVid) {
        int oldVid = new2old[newVid];
        conNew[newVid].reserve(con[oldVid].size());
        for (int i = 0; i < con[oldVid].size(); ++i) {
            unsigned oldAdj = con[oldVid][i];
            conNew[newVid].push_back((unsigned)old2new[oldAdj]);
        }
        sort(conNew[newVid].begin(), conNew[newVid].end());
    }
    con.swap(conNew);

    if (!label.empty() && !pos.empty()) {
        ArrayOnHeap<vector<unsigned>> labelNew(totalV);
        ArrayOnHeap<vector<int>> posNew(totalV);

        for (int newVid = 0; newVid < totalV; ++newVid) {
            int oldVid = new2old[newVid];
            posNew[newVid] = pos[oldVid];

            labelNew[newVid].reserve(label[oldVid].size());
            for (int i = 0; i < label[oldVid].size(); ++i) {
                unsigned oldHub = label[oldVid][i] >> MAXMOV;
                unsigned dis = label[oldVid][i] & MASK;
                unsigned newHub = (unsigned)old2new[oldHub];
                labelNew[newVid].push_back((newHub << MAXMOV) | dis);
            }

            for (int d = 0; d < posNew[newVid].size(); ++d) {
                int l = d == 0 ? 0 : posNew[newVid][d - 1];
                int r = posNew[newVid][d];
                sort(labelNew[newVid].begin() + l, labelNew[newVid].begin() + r);
            }
        }

        label = std::move(labelNew);
        pos = std::move(posNew);
    }

    v2p = new_v2p;
    p2v = new_p2v;

    // #region agent log
    if (!label.empty()) {
        int maxHub = 0;
        for (int u = 0; u < totalV; ++u)
            for (auto l : label[u]) {
                int hub = (int)(l >> MAXMOV);
                if (hub >= totalV || hub < 0) {
                    DBG_LOG("A", "GraphReorder_vp:end", "OOB hub after reorder",
                            "\"data\":{\"u\":%d,\"hub\":%d,\"totalV\":%d}", u, hub, totalV);
                }
                if (hub > maxHub) maxHub = hub;
            }
        DBG_LOG("A", "GraphReorder_vp:end", "max hub ID check", "\"data\":{\"maxHub\":%d,\"totalV\":%d}", maxHub,
                totalV);
    }
    // #endregion
}

bool can_update(int v, int dis, char* nowdis) {
    for (int i = 0; i < (int)label[v].size(); ++i) {
        int w = label[v][i] >> MAXMOV, d = label[v][i] & MASK;
        if (nowdis[w] >= 0 && nowdis[w] + d <= dis) return false;
    }
    return true;
}

bool can_update_add(int v, int dis, char* nowdis) {
    int dmax = pos[v].size() - 1, pla = dis > dmax ? pos[v][dmax] : pos[v][dis];

    for (int i = 0; i < pla; ++i) {
        int w = label[v][i] >> MAXMOV, d = label[v][i] & MASK;
        if (nowdis[w] >= 0 && nowdis[w] + d <= dis) return false;
    }

    for (int i = 0; i < clab[v].size(); ++i) {
        int w = clab[v][i] >> MAXMOV, d = clab[v][i] & MASK;
        if (nowdis[w] >= 0 && nowdis[w] + d <= dis) return false;
    }

    return true;
}

bool can_update_delete(int v, int dis, char* nowdis) {
    int pla1 = dis < pos[v].size() ? pos[v][dis] : label[v].size();
    for (int i = 0; i < pla1; ++i) {
        int w = label[v][i] >> MAXMOV, d = label[v][i] & MASK;
        if (nowdis[w] >= 0 && nowdis[w] + d <= dis) return false;
    }

    return true;
}

void IndexBuild() {
    cout << "Execute PSL to construct the 2-hop index ......" << endl;
    omp_set_num_threads(threads);

    pos = ArrayOnHeap<vector<int>>(totalV);

    label = ArrayOnHeap<vector<unsigned>>(totalV);

    for (int i = 0; i < totalV; ++i) {
        label[i].push_back((((unsigned)i) << MAXMOV) | 0);

        for (int j = 0; j < con[i].size() && con[i][j] < i; ++j) {
            label[i].push_back((((unsigned)con[i][j]) << MAXMOV) | 1);
        }

        pos[i].push_back(1), pos[i].push_back(label[i].size());
    }

    int dis = 2;
    for (long long cnt = 1; cnt && dis <= MAXDIS; ++dis) {
        cnt = 0;
        ArrayOnHeap<vector<unsigned>> label_new(totalV);
        double t1 = omp_get_wtime();
#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            ArrayOnHeap<unsigned char> used(totalV / 8 + 1);
            vector<int> cand;

            ArrayOnHeap<char> nowdis(totalV, ArrayOnHeap<char>::uninitialized);
            nowdis.memset(-1);

            for (int u = pid; u < totalV; u += np) {
                cand.clear();

                for (int i = 0; i < con[u].size(); ++i) {
                    int w = con[u][i];

                    for (int j = pos[w][dis - 2]; j < pos[w][dis - 1]; ++j) {
                        int v = label[w][j] >> MAXMOV;
                        if (v >= u) {
                            break;
                            // continue;
                        }

                        if (!(used[v / 8] & (1 << (v % 8)))) {
                            used[v / 8] |= (1 << (v % 8)), cand.push_back(v);
                        }
                    }
                }

                int n_cand = 0;
                for (int i = 0; i < (int)label[u].size(); ++i) nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;

                for (int i = 0; i < (int)cand.size(); ++i) {
                    used[cand[i] / 8] = 0;
                    if (can_update(cand[i], dis, nowdis.data())) cand[n_cand++] = cand[i];
                }

                cand.resize(n_cand);
                sort(cand.begin(), cand.end());

                for (int i = 0; i < (int)cand.size(); ++i) {
                    label_new[u].push_back((((unsigned)cand[i]) << MAXMOV) | (unsigned)dis), ++local_cnt;
                }
                for (int i = 0; i < (int)label[u].size(); ++i) nowdis[label[u][i] >> MAXMOV] = -1;
            }

#pragma omp critical
            {
                cnt += local_cnt;
            }
        }
        // double t2 = omp_get_wtime();
#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            for (int u = pid; u < totalV; u += np) {
                label[u].insert(label[u].end(), label_new[u].begin(), label_new[u].end());

                vector<unsigned>(label[u]).swap(label[u]);

                vector<unsigned>().swap(label_new[u]);

                pos[u].push_back(label[u].size());
            }
        }
        double t3 = omp_get_wtime();

        cout << "Distance: " << dis << "   Cnt: " << cnt << "time = " << t3 - t1 << endl;
    }
}

void IndexSave(string path) {
    cout << "Store the 2-hop index ......" << endl;

    FILE* fout = fopen(path.c_str(), "wb");

    fwrite(&totalV, sizeof(int), 1, fout);  // 顶点数

    for (int i = 0; i < totalV; ++i) {
        int len = (int)label[i].size();
        fwrite(&len, sizeof(int), 1, fout);  // 每个顶点的 labels 的 数量
    }

    ArrayOnHeap<unsigned> s(totalV, ArrayOnHeap<unsigned>::uninitialized);

    for (int i = 0; i < totalV; ++i) {
        int len = (int)label[i].size();
        for (int j = 0; j < len; ++j) s[j] = label[i][j];
        fwrite(s.data(), sizeof(unsigned), len, fout);
    }

    // ===========================
    for (int i = 0; i < totalV; ++i) {
        int len = (int)pos[i].size();
        fwrite(&len, sizeof(int), 1, fout);
    }

    for (int i = 0; i < totalV; ++i) {
        int len = (int)pos[i].size();
        for (int j = 0; j < len; ++j) s[j] = pos[i][j];
        fwrite(s.data(), sizeof(unsigned), len, fout);
    }

    fwrite(&MAXMOV, sizeof(int), 1, fout);
    fclose(fout);
}

void IndexWrite() {
    for (int i = 0; i < totalV; ++i) {
        std::cout << std::format("L[{}] = ", i);
        int len = (int)label[i].size();
        for (int j = 0; j < len; ++j) {
            std::cout << std::format("({}, {}) ", label[i][j] >> MAXMOV, label[i][j] & MASK);
        }
        std::cout << std::endl;
    }
}

void IndexLoad(string path) {
    FILE* fin = fopen(path.c_str(), "rb");
    size_t elements_read = fread(&totalV, sizeof(int), 1, fin);

    ArrayOnHeap<int> len(totalV, ArrayOnHeap<int>::uninitialized);
    elements_read = fread(len.data(), sizeof(int), totalV, fin);

    label = ArrayOnHeap<vector<unsigned>>(totalV);
    ArrayOnHeap<unsigned> s(totalV, ArrayOnHeap<unsigned>::uninitialized);

    for (int i = 0; i < totalV; ++i) {
        assert(len[i] <= totalV);
        elements_read = fread(s.data(), sizeof(unsigned), len[i], fin);
        label[i].reserve(len[i]);
        label[i].assign(s.data(), s.data() + len[i]);
    }

    elements_read = fread(len.data(), sizeof(int), totalV, fin);
    pos = ArrayOnHeap<vector<int>>(totalV);
    for (int i = 0; i < totalV; ++i) {
        assert(len[i] <= totalV);
        elements_read = fread(s.data(), sizeof(unsigned), len[i], fin);
        pos[i].reserve(len[i]);
        pos[i].assign(s.data(), s.data() + len[i]);
    }

    elements_read = fread(&MAXMOV, sizeof(unsigned), 1, fin);
    MAXDIS = 1 << MAXMOV;
    MASK = MAXDIS - 1;

    // #region agent log
    fclose(fin);
    int maxHub = 0, minPosSize = totalV, maxPosSize = 0;
    for (int u = 0; u < totalV; ++u) {
        for (auto l : label[u]) {
            int hub = (int)(l >> MAXMOV);
            if (hub >= totalV || hub < 0)
                DBG_LOG("A", "IndexLoad:end", "OOB hub after load", "\"data\":{\"u\":%d,\"hub\":%d,\"totalV\":%d}", u,
                        hub, totalV);
            if (hub > maxHub) maxHub = hub;
        }
        int ps = (int)pos[u].size();
        if (ps < minPosSize) minPosSize = ps;
        if (ps > maxPosSize) maxPosSize = ps;
    }
    DBG_LOG("AB", "IndexLoad:end", "post-load validation",
            "\"data\":{\"totalV\":%d,\"maxHub\":%d,\"minPosSize\":%d,\"maxPosSize\":%d,\"MAXMOV\":%u}", totalV, maxHub,
            minPosSize, maxPosSize, MAXMOV);
    // #endregion
}

void IndexPrint() {
    // string new_filename = "Indo_PSL.txt";
    // const char *file = new_filename.c_str();
    // fstream outfileX;
    // outfileX.open(file, ios::out);

    for (int i = 0; i < totalV; ++i) {
        vector<unsigned>& lab = label[i];
        // cout<<"id: "<<i<<"      ";
        for (int j = 0; j < lab.size(); ++j) {
            unsigned vid = lab[j] >> MAXMOV, dis = lab[j] & MASK;
            cout << "$(v_" << vid << "," << dis << ")$, ";
        }
        cout << endl;
        // vector<unsigned>& cl = clab[i];
        // for (int j=0; j<cl.size(); ++j){
        //     unsigned vid = cl[j] >> MAXMOV, dis = cl[j] & MASK;
        //     cout<<"$("<<vid<<","<<dis<<")$, ";
        // }

        // outfileX<<lab.size()<<endl;
        // for (int j=0; j<pos[i].size(); ++j){
        //     cout<<pos[i][j]<<"  ";
        // }
        // cout<<endl;
    }
    // outfileX.close();
}

void IndexSize(int flg) {
    long long cntt = 0;

    for (int i = 0; i < totalV; ++i) {
        cntt += label[i].size();
        if (flg == 1) cntt += clab[i].size();
    }

    cout << "Total label number: " << cntt << endl;
}

void labelCheck() {
    // 主要检查label和pos的格式
    int dis = pos[0].size();

    for (int i = 0; i < totalV; ++i) {
        vector<unsigned>& lab = label[i];

        for (int dd = 1; dd < dis; ++dd) {
            for (int pp = pos[i][dd - 1]; pp < pos[i][dd] - 1; ++pp) {
                unsigned v1 = lab[pp] >> MAXMOV, d1 = lab[pp] & MASK;
                unsigned v2 = lab[pp + 1] >> MAXMOV, d2 = lab[pp + 1] & MASK;

                if (v1 >= v2) cout << "error!" << endl;
            }
        }
    }

    for (int i = 0; i < totalV; ++i) {
        vector<unsigned>& lab = label[i];
        for (int dd = 1; dd < dis; ++dd)
            for (int pp = pos[i][dd - 1]; pp < pos[i][dd]; ++pp) {
                unsigned v1 = lab[pp] >> MAXMOV, d1 = lab[pp] & MASK;
                if (d1 != dd) cout << "error!" << endl;
            }
    }
}

int CurDis(unsigned src, unsigned tgt, unsigned dcur) {
    unsigned elem = src << MAXMOV | (dcur - 1);

    for (int i = 0; i < con[tgt].size(); ++i) {
        // con[tgt][i]<src 的情况可以不考虑, check (src, dcur-1) 是否属于 L(con[tgt][i])
        unsigned vid = con[tgt][i];

        if (vid < src) continue;

        int pla1 = dcur == 1 ? 0 : pos[vid][dcur - 2], pla2 = pos[vid][dcur - 1];

        for (int j = pla1; j < pla2; ++j)
            if (label[vid][j] == elem) return 1;
    }

    return 0;
}

// ============ for edge deletion =============

void DeleteGraphRandByEdge(int deleteEdgeNum) {
    vector<pair<int, int>> allEdges;
    allEdges.reserve(totalV);

    for (int vid = 0; vid < totalV; ++vid) {
        for (int i = 0; i < con[vid].size(); ++i) {
            int adj = con[vid][i];
            if (vid < adj) allEdges.push_back(make_pair(vid, adj));
        }
    }

    if (allEdges.empty() || deleteEdgeNum <= 0) return;

    int needDelete = min(deleteEdgeNum, (int)allEdges.size());
    mt19937 rng((unsigned)time(NULL));
    shuffle(allEdges.begin(), allEdges.end(), rng);

    for (int i = 0; i < needDelete; ++i) {
        int vid = allEdges[i].first, adj = allEdges[i].second;

        conD[vid].push_back(adj);  // for the index update
        conD[adj].push_back(vid);

        con[vid].erase(remove(con[vid].begin(), con[vid].end(), adj), con[vid].end());
        con[adj].erase(remove(con[adj].begin(), con[adj].end(), vid), con[adj].end());
    }
}

void DeleteGraph(float cc) {
    if (cc < 0) {
        DeleteGraphRandByEdge(Dcnt);
        return;
    }
    for (int i = 0; i < Dcnt; ++i) {
        int vid = (int)(totalV * cc) + i;
        int c = con[vid].size(), pla = (int)(c * 0.9),
            adj = con[vid][pla];  // <i,adj>

        // vid = 0, adj = 4;
        // cout<<"delete: "<<vid<<"   "<<adj<<endl;

        conD[vid].push_back(adj);  // for the index update
        conD[adj].push_back(vid);

        con[vid].erase(remove(con[vid].begin(), con[vid].end(), adj), con[vid].end());
        con[adj].erase(remove(con[adj].begin(), con[adj].end(), vid), con[adj].end());
    }
}

void DeleteEdge(int vid, int adj) {
    conD[vid].push_back(adj);  // for the index update
    conD[adj].push_back(vid);

    con[vid].erase(remove(con[vid].begin(), con[vid].end(), adj), con[vid].end());
    con[adj].erase(remove(con[adj].begin(), con[adj].end(), vid), con[adj].end());
}

void IndexDel_Parallel() {  // 并行删除error label

    omp_set_num_threads(threads);

    vaff.resize(totalV, -1);

    delete_labels = ArrayOnHeap<vector<unsigned>>(totalV);

    int dis = 1;
    for (long long cnt = 1; cnt && dis <= MAXDIS; ++dis) {
        cnt = 0;
        ArrayOnHeap<vector<unsigned>> label_new(totalV);

#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            ArrayOnHeap<unsigned char> used(totalV);
            vector<int> cand;

            ArrayOnHeap<int> nowdis(totalV, ArrayOnHeap<int>::uninitialized);
            nowdis.memset(-1);

            for (int u = pid; u < totalV; u += np) {
                cand.clear();

                int n_cand = 0, pla1 = pos[u][dis - 1];
                int pla2 = dis < pos[u].size() ? pos[u][dis] : label[u].size();

                for (int i = pla1; i < pla2; ++i) nowdis[label[u][i] >> MAXMOV] = dis;
                // =========================================================================

                for (int i = 0; i < con[u].size(); ++i) {  // check the delete_label of neighbors

                    int w = con[u][i];

                    for (int j = 0; j < delete_labels[w].size(); ++j) {
                        int v = delete_labels[w][j] >> MAXMOV;

                        if (v >= u) break;  // 已经排过序的

                        if (nowdis[v] != dis) continue;  // = (cand[i], dis) 不包含在 label[u] 中 =

                        if (used[v] == 0) {
                            used[v] = 1, cand.push_back(v);
                        }
                    }
                }

                // ====================================================

                for (int i = 0; i < conD[u].size(); ++i) {  // check the label of deleted neighbors

                    int w = conD[u][i];

                    int pp1 = dis == 1 ? 0 : pos[w][dis - 2], pp2 = pos[w][dis - 1];

                    for (int j = pp1; j < pp2; ++j) {
                        int v = label[w][j] >> MAXMOV;

                        if (v >= u) break;

                        if (nowdis[v] != dis) continue;  // = (cand[i], dis) 不包含在 label[u] 中 =

                        if (used[v] == 0) {
                            used[v] = 1, cand.push_back(v);
                        }
                    }

                    for (int j = 0; j < delete_labels[w].size(); ++j) {
                        int v = delete_labels[w][j] >> MAXMOV;

                        if (v >= u) break;  // 已经排过序的

                        if (nowdis[v] != dis) continue;  // = (cand[i], dis) 不包含在 label[u] 中 =

                        if (used[v] == 0) {
                            used[v] = 1, cand.push_back(v);
                        }
                    }
                }

                // =========================================================================

                for (int i = 0; i < (int)cand.size(); ++i) {
                    used[cand[i]] = 0;

                    int D_now = CurDis(cand[i], u, dis);

                    if (D_now == 1) continue;  // 不需要删除

                    cand[n_cand++] = cand[i];
                }

                cand.resize(n_cand);

                sort(cand.begin(), cand.end());

                if (cand.size() > 0) {
                    if (vaff[u] == -1)
                        vaff[u] = (int)cand[0];
                    else
                        vaff[u] = min(vaff[u], (int)cand[0]);  // 后续剪枝

                    // cout<<"id: "<<u<<"    ";
                    // for (int ii=0; ii<cand.size(); ++ii){
                    //     cout<<"v_"<<cand[ii]<<"  ";
                    // }
                    // cout<<endl;
                }

                for (int i = 0; i < (int)cand.size(); ++i) {
                    label_new[u].push_back((((unsigned)cand[i]) << MAXMOV) | (unsigned)dis), ++local_cnt;
                }

                for (int i = pla1; i < pla2; ++i) nowdis[label[u][i] >> MAXMOV] = -1;
            }

#pragma omp critical
            {
                cnt += local_cnt;
            }
        }

#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();

            for (int u = pid; u < totalV; u += np) {
                delete_labels[u].clear();
                delete_labels[u].insert(delete_labels[u].end(), label_new[u].begin(), label_new[u].end());
                vector<unsigned>(delete_labels[u]).swap(delete_labels[u]);
                vector<unsigned>().swap(label_new[u]);

                for (int i = 0; i < delete_labels[u].size(); ++i) {
                    label[u].erase(remove(label[u].begin(), label[u].end(), delete_labels[u][i]), label[u].end());
                    unsigned dd = delete_labels[u][i] & MASK;
                    for (int ij = dd; ij < pos[u].size(); ++ij) pos[u][ij] -= 1;
                }
            }
        }

        cout << "Distance: " << dis << "  Delete Cnt: " << cnt << endl;
    }

    // for (int i=0; i<totalV; ++i){
    //     cout<<vaff[i]<<endl;
    // }
}

void IndexDel_Add() {
    omp_set_num_threads(threads);

    clab = ArrayOnHeap<vector<unsigned>>(totalV);

#pragma omp parallel
    {
        int pid = omp_get_thread_num(), np = omp_get_num_threads();
        for (int u = pid; u < totalV; u += np) {
            for (int i = pos[u][0]; i < pos[u][1]; ++i) {
                unsigned vid = label[u][i] >> MAXMOV;

                if (vaff[vid] == -1) continue;

                clab[u].push_back(vid);
            }
        }
    }

    int dis = 2;
    for (long long cnt = 1; cnt > 0; ++dis) {
        cnt = 0;
        ArrayOnHeap<vector<unsigned>> label_new(totalV);

#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0, local_cnt1 = 0;
            ArrayOnHeap<unsigned char> used(totalV / 8 + 1);
            vector<int> cand;

            ArrayOnHeap<char> nowdis(totalV, ArrayOnHeap<char>::uninitialized);
            nowdis.memset(-1);

            for (int u = pid; u < totalV; u += np) {  // 可以考虑减少线程调度的开销，从代码的层次

                cand.clear();

                for (int i = 0; i < con[u].size(); ++i) {
                    int w = con[u][i];

                    for (int j = 0; j < clab[w].size(); ++j) {
                        int v = clab[w][j];  // clab中包含所有需要考虑的vaff点

                        if (v >= u) break;  // rank 剪枝

                        if (vaff[u] == -1 and vaff[v] == -1) continue;

                        if (!(used[v / 8] & (1 << (v % 8)))) {
                            used[v / 8] |= (1 << (v % 8)), cand.push_back(v);
                        }
                    }

                    if (vaff[u] != -1) {  // vaff点需要遍历更多的候选label

                        for (int j = pos[w][dis - 2]; j < pos[w][dis - 1]; ++j) {
                            int v = label[w][j] >> MAXMOV;

                            if (v >= u) break;  // rank 剪枝

                            if (v < vaff[u]) continue;

                            if (!(used[v / 8] & (1 << (v % 8)))) {
                                used[v / 8] |= (1 << (v % 8)), cand.push_back(v);
                            }
                        }
                    }
                }

                int n_cand = 0;
                if (cand.size() == 0) continue;

                int pla1 = dis < pos[u].size() ? pos[u][dis] : label[u].size();

                for (int i = 0; i < pla1; ++i)  // 便于判断
                    nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;

                for (int i = 0; i < (int)cand.size(); ++i) {
                    used[cand[i] / 8] = 0;

                    if (nowdis[cand[i]] == dis) {
                        ++local_cnt;  // 直连标签在下一个回合中，需要进行判断
                        if (vaff[cand[i]] != -1) cand[n_cand++] = cand[i];
                    } else if (can_update_delete(cand[i], dis, nowdis.data())) {
                        // cout<<"id: "<<u<<"  v_"<<cand[i]<<endl;
                        cand[n_cand++] = cand[i], ++local_cnt;
                    }
                }

                cand.resize(n_cand);
                sort(cand.begin(), cand.end());

                for (int i = 0; i < (int)cand.size(); ++i) {
                    label_new[u].push_back(cand[i]);
                }

                for (int i = 0; i < pla1; ++i)  // 便于判断
                    nowdis[label[u][i] >> MAXMOV] = -1;
            }

#pragma omp critical
            {
                cnt += local_cnt;
            }
        }

#pragma omp parallel
        {
            // Update labels and pos, 这里的insert方式要修改
            int pid = omp_get_thread_num(), np = omp_get_num_threads();

            for (int u = pid; u < totalV; u += np) {
                vector<unsigned>& lab = label[u];
                vector<int>& pp = pos[u];
                clab[u].clear();

                if (dis < pp.size()) {  // 还是在现有的label之间

                    int p1 = pp[dis - 1];

                    for (int ii = 0; ii < label_new[u].size(); ++ii) {
                        unsigned vid = label_new[u][ii], elem = (vid << MAXMOV) | dis;
                        int pla = -1, k;
                        clab[u].push_back(vid);

                        for (k = p1; k < pp[dis]; ++k) {
                            if (lab[k] == elem) {
                                pla = -2;
                                break;  // 不需要更新到labels中去  and vaff[vid] != -1
                            } else if (lab[k] > elem) {
                                pla = k;
                                break;
                            }
                        }

                        if (pla == -2) continue;

                        if (pla == -1) pla = k;

                        p1 = pla;

                        lab.insert(lab.begin() + pla, elem);

                        for (int ii = dis; ii < pp.size(); ++ii) pp[ii] += 1;
                    }
                } else {
                    for (int i = 0; i < label_new[u].size(); ++i) {
                        unsigned elem = label_new[u][i] << MAXMOV | dis;
                        lab.push_back(elem);
                        clab[u].push_back(label_new[u][i]);
                    }
                    pp.push_back(lab.size());
                }

                vector<unsigned>(clab[u]).swap(clab[u]);
                vector<unsigned>().swap(label_new[u]);
            }
        }
        cout << "dis: " << dis << "  cnt: " << cnt << endl;
    }
}

void merge_labels(const ArrayOnHeap<vector<unsigned>>& new_label) {
#pragma omp parallel
    {
        int pid = omp_get_thread_num(), np = omp_get_num_threads();
        for (int u = pid; u < totalV; u += np) {
            if (new_label[u].empty()) continue;

            vector<unsigned> merged;
            merged.reserve(label[u].size() + new_label[u].size());

            int i = 0, j = 0;
            int li = (int)label[u].size(), lj = (int)new_label[u].size();
            while (i < li && j < lj) {
                unsigned da = label[u][i] & MASK, va = label[u][i] >> MAXMOV;
                unsigned db = new_label[u][j] & MASK, vb = new_label[u][j] >> MAXMOV;
                if (da < db || (da == db && va <= vb)) {
                    merged.push_back(label[u][i++]);
                } else {
                    merged.push_back(new_label[u][j++]);
                }
            }
            while (i < li) merged.push_back(label[u][i++]);
            while (j < lj) merged.push_back(new_label[u][j++]);

            pos[u].clear();
            unsigned prevDis = 0;
            for (int k = 0; k < (int)merged.size(); ++k) {
                unsigned d = merged[k] & MASK;
                while (prevDis < d) {
                    pos[u].push_back(k);
                    ++prevDis;
                }
            }
            pos[u].push_back((int)merged.size());

            label[u].swap(merged);
        }
    }
}

//* ======= for reorder ========
void IndexReorder() {
    omp_set_num_threads(threads);

    vaff.resize(totalV, -1);

    ArrayOnHeap<vector<unsigned>> bad_inv_labels(totalV);
    if (label.empty() || pos.empty()) {
        std::cout << "No label to reorder" << std::endl;
        return;
    }

    //* ===================== removing part ==========================

    ArrayOnHeap<vector<unsigned>> prev_mod_labels(totalV), cur_mod_labels(totalV);
    ArrayOnHeap<vector<size_t>> deleteIndexes(totalV);

    unsigned reorderMaxDis = 0;
    for (int u = 0; u < totalV; ++u) {
        reorderMaxDis = std::max(reorderMaxDis, label[u].back() & MASK);
    }

    // #region agent log
    DBG_LOG("B", "IndexReorder:start", "reorderMaxDis and pos check",
            "\"data\":{\"reorderMaxDis\":%u,\"totalV\":%d,\"posSize0\":%d}", reorderMaxDis, totalV, (int)pos[0].size());
    for (int u = 0; u < totalV; ++u) {
        if ((int)pos[u].size() <= (int)reorderMaxDis - 1) {
            DBG_LOG("B", "IndexReorder:start", "pos too small for reorderMaxDis",
                    "\"data\":{\"u\":%d,\"posSize\":%d,\"reorderMaxDis\":%u}", u, (int)pos[u].size(), reorderMaxDis);
            break;
        }
        for (auto l : label[u]) {
            int hub = (int)(l >> MAXMOV);
            if (hub < 0 || hub >= totalV) {
                DBG_LOG("A", "IndexReorder:start", "OOB hub before reorder loop",
                        "\"data\":{\"u\":%d,\"hub\":%d,\"totalV\":%d}", u, hub, totalV);
                break;
            }
        }
    }
    // #endregion

    long long cur_cnt = 0, prev_cnt = 0;
    std::cout << "=== Reorder starts ===" << std::endl;

    double t1 = omp_get_wtime();

    for (int dis = 1; dis <= reorderMaxDis; ++dis) {
#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            ArrayOnHeap<bool> used(totalV);
            vector<int> cand;
            vector<pair<int, unsigned>> local_bad_inv;

            ArrayOnHeap<int> nowIndex(totalV, ArrayOnHeap<int>::uninitialized);
            nowIndex.memset(-1);

            for (int u = pid; u < totalV; u += np) {
                cand.clear();

                int pla1 = (dis - 1 < pos[u].size()) ? pos[u][dis - 1] : label[u].size();
                int pla2 = dis < pos[u].size() ? pos[u][dis] : label[u].size();
                for (int i = pla1; i < pla2; ++i) {
                    int v = label[u][i] >> MAXMOV;
                    nowIndex[v] = i;
                    // assert(v > u);
                    if (v > u) {                                           //* v没有u重要，删除该label
                        bad_inv_labels[v].push_back((u << MAXMOV) | dis);  // TODO: need locking
                        deleteIndexes[u].push_back(i);
                        if (!used[v]) {
                            used[v] = true;
                            cand.push_back(v);
                        }
                    }
                }
                // =========================================================================
                if (prev_cnt > 0) {                            //* 如果上一层没删除label,就不会推到本层
                    for (int i = 0; i < con[u].size(); ++i) {  // check the delete_label of neighbors

                        int w = con[u][i];

                        for (size_t j = 0; j < prev_mod_labels[w].size(); ++j) {  //* 只包含dis-1的被删除的labels
                            int v = prev_mod_labels[w][j] >> MAXMOV, pd = (prev_mod_labels[w][j] & MASK);
                            if (pd != dis - 1) {
                                std::cout << std::format("dis = {}, prev[{}][{}].dis = {}, size = {}", dis, w, j, pd,
                                                         prev_mod_labels[w].size())
                                          << std::endl;
                                assert(0);
                            }

                            if (nowIndex[v] == -1) continue;  //* = (cand[i], dis) 不包含在 label[u] 中 =

                            if (!used[v]) {
                                used[v] = true;
                                cand.push_back(v);
                                deleteIndexes[u].push_back(nowIndex[v]);
                            }
                        }
                    }
                }

                for (int i = 0; i < (int)cand.size(); ++i) {
                    used[cand[i]] = false;
                    cur_mod_labels[u].push_back((((unsigned)cand[i]) << MAXMOV) | (unsigned)dis), ++local_cnt;
                }

                for (int i = pla1; i < pla2; ++i) nowIndex[label[u][i] >> MAXMOV] = -1;
            }

#pragma omp critical
            {
                cur_cnt += local_cnt;
                // for (auto& [v, data] : local_bad_inv) bad_inv_labels[v].push_back(data);
            }
        }
#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            for (int u = pid; u < totalV; u += np) {
                prev_mod_labels[u].swap(cur_mod_labels[u]);
                cur_mod_labels[u].clear();
            }
        }

        cout << "Distance: " << dis << "  Delete Cnt: " << cur_cnt << endl;

        // #region agent log
        DBG_LOG("C", "IndexReorder:removing", "end of dis iteration", "\"data\":{\"dis\":%d,\"cur_cnt\":%lld}", dis,
                cur_cnt);
        // #endregion

        prev_cnt = cur_cnt;
        cur_cnt = 0;
    }

    // #region agent log
    DBG_LOG("C", "IndexReorder:afterRemovingLoop", "removing loop done", "\"data\":{}");
    // #endregion

    double t2 = omp_get_wtime();
    std::cout << "get delete =" << t2 - t1 << "s" << std::endl;

    //* 删除不需要的index
    size_t total_deleted_cnt = 0;
#pragma omp parallel
    {
        int pid = omp_get_thread_num(), np = omp_get_num_threads();
        size_t local_deleted_cnt = 0;
        for (int u = pid; u < totalV; u += np) {
            local_deleted_cnt += deleteIndexes[u].size();
            if (deleteIndexes[u].empty()) continue;

            sort(deleteIndexes[u].begin(), deleteIndexes[u].end());
            vaff[u] = deleteIndexes[u][0];

            size_t write = 0, di = 0, delN = deleteIndexes[u].size();
            for (size_t i = 0; i < label[u].size(); ++i) {
                if (di < delN && deleteIndexes[u][di] == i) {
                    ++di;
                } else {
                    label[u][write++] = label[u][i];
                }
            }
            label[u].resize(write);

            pos[u].clear();
            if (label[u].empty()) continue;
            int maxDisOnU = label[u].back() & MASK;
            size_t tail = 0;
            for (int d = 0; d < maxDisOnU; ++d) {
                while (tail != label[u].size() && (label[u][tail] & MASK) <= d) ++tail;
                pos[u].push_back(tail);
                if (tail == label[u].size()) break;
            }
        }
#pragma omp critical
        {
            total_deleted_cnt += local_deleted_cnt;
        }
    }
    double t3 = omp_get_wtime();
    std::cout << "deleting =" << t3 - t2 << "s, total delete = " << total_deleted_cnt << std::endl;

    // for (int u = 0; u < totalV; ++u) {
    //     int prev = 0;
    //     for (int d = 0; d < (int)pos[u].size(); ++d) {
    //         assert(pos[u][d] >= prev && pos[u][d] <= (int)label[u].size());
    //         for (int j = prev; j < pos[u][d]; ++j) {
    //             unsigned actualDis = label[u][j] & MASK;
    //             assert((label[u][j] >> MAXMOV) <= u);
    //             assert(actualDis == (unsigned)d && "label dis does not match pos segment");
    //         }
    //         prev = pos[u][d];
    //     }
    //     int lastD = (int)pos[u].size();
    //     for (int j = prev; j < (int)label[u].size(); ++j) {
    //         unsigned actualDis = label[u][j] & MASK;
    //         assert((label[u][j] >> MAXMOV) <= u);
    //         assert(actualDis == (unsigned)lastD && "label dis does not match last segment");
    //     }
    // }
    // cout << "pos check after removing part: PASS" << endl;

    // #region agent log
    DBG_LOG("C", "IndexReorder:afterDeletion", "deletion done, entering insert", "\"data\":{}");
    // #endregion

    //* ====================== inserting part ===========================
    ArrayOnHeap<int> badLabelTail(totalV);

    // ArrayOnHeap<vector<unsigned>> new_label(totalV);

    size_t total_new_cnt = 0;
    cur_cnt = 0, prev_cnt = 0;

    for (unsigned dis = 1; dis <= reorderMaxDis || prev_cnt != 0; ++dis) {
        long long cand_size = 0, enum_size = 0;
        double t1 = omp_get_wtime();
#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            // int pid = 0, np = 1;
            long long local_cnt = 0, local_cand_size = 0, local_enum_size = 0, local_prev = 0, local_bad = 0;
            ArrayOnHeap<bool> used(totalV);
            ArrayOnHeap<char> nowdis(totalV, ArrayOnHeap<char>::uninitialized);
            nowdis.memset(-1);
            vector<int> cand;

            for (int u = pid; u < totalV; u += np) {
                cand.clear();
                //* add bad labels of current dis into cand
                while (badLabelTail[u] != bad_inv_labels[u].size() &&
                       (bad_inv_labels[u][badLabelTail[u]] & MASK) <= dis) {
                    int v = bad_inv_labels[u][badLabelTail[u]] >> MAXMOV,
                        labelDis = bad_inv_labels[u][badLabelTail[u]] & MASK;
                    if (labelDis == dis) {
                        used[v] = true;
                        cand.push_back(v);
                    } else {
                        // assert(0);
                    }
                    ++badLabelTail[u];
                    ++local_bad;
                }

                int pla1 = dis < pos[u].size() ? pos[u][dis] : label[u].size();

                //* 从上一层被加入的label往下推
                if (prev_cnt > 0) {
                    int pla2 = (dis + 1 < pos[u].size()) ? pos[u][dis + 1] : label[u].size();
                    for (size_t i = pla1; i < pla2; ++i) {
                        used[label[u][i] >> MAXMOV] = true;
                    }
                    for (int w : con[u]) {
                        for (unsigned l : prev_mod_labels[w]) {
                            int v = l >> MAXMOV;
                            if (v >= u) {
                                // break;
                                continue;
                            }
                            // assert((l & MASK) == dis - 1);
                            if (!used[v]) {
                                used[v] = true;
                                cand.push_back(v);
                            }
                        }
                        local_prev += prev_mod_labels[w].size();
                    }
                    for (size_t i = pla1; i < pla2; ++i) {
                        used[label[u][i] >> MAXMOV] = false;
                    }
                }

                if (cand.size() == 0) continue;

                // std::cout << "dis = " << dis << ", cand size = " << cand.size() << std::endl;
                local_cand_size += cand.size();

                for (int i = 0; i < pla1; ++i) {  // 便于判断
                    if ((label[u][i] >> MAXMOV) > u) continue;
                    nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;
                }
                // for (unsigned l : new_label[u]) {
                //     if ((l >> MAXMOV) > u) continue;
                //     nowdis[l >> MAXMOV] = l & MASK;
                // }

                auto can_update_delete_with_new = [&](int v, int dis) {
                    int pla1 = dis < pos[v].size() ? pos[v][dis] : label[v].size();
                    bool ok = false;
                    for (int i = 0; i < pla1; ++i) {
                        int w = label[v][i] >> MAXMOV, d = label[v][i] & MASK;
                        if (d >= dis) {
                            ok = true;
                            local_enum_size += i + 1;
                            break;
                        }
                        if (nowdis[w] >= 0 && nowdis[w] + d <= dis) {
                            // if (u == 1756 && v == 1217 && dis == 3) {
                            //     std::cout << "WA: prune hub = " << w << ", dis = " << d << std::endl;
                            // }
                            return false;
                        }
                    }
                    if (!ok) local_enum_size += pla1;
                    // ok = false;
                    // for (int i = 0; i < (int)new_label[v].size(); ++i) {
                    //     int w = new_label[v][i] >> MAXMOV, d = new_label[v][i] & MASK;
                    //     if (d >= dis) {
                    //         ok = true;
                    //         local_enum_size += i + 1;
                    //         break;
                    //     }
                    //     if (nowdis[w] >= 0 && nowdis[w] + d <= dis) {
                    //         return false;
                    //     }
                    // }
                    // if (!ok) local_enum_size += new_label[v].size();

                    return true;
                };

                for (int i = 0; i < (int)cand.size(); ++i) {
                    used[cand[i]] = 0;
                    if (vaff[cand[i]] == -1 && (vaff[u] == -1 || vaff[u] > cand[i])) continue;
                    if (nowdis[cand[i]] == -1 && can_update_delete_with_new(cand[i], dis)) {
                        // cout<<"id: "<<u<<"  v_"<<cand[i]<<endl;
                        ++local_cnt;
                        unsigned labelU = (((unsigned)cand[i]) << MAXMOV) | (unsigned)dis;
                        cur_mod_labels[u].push_back(labelU);
                    }
                }

                for (int i = 0; i < pla1; ++i)  // 便于判断
                    nowdis[label[u][i] >> MAXMOV] = -1;
                // for (unsigned l : new_label[u]) {
                //     nowdis[l >> MAXMOV] = -1;
                // }

                std::sort(cur_mod_labels[u].begin(), cur_mod_labels[u].end());
                // new_label[u].insert(new_label[u].end(), cur_mod_labels[u].begin(), cur_mod_labels[u].end());
                if (!cur_mod_labels[u].empty()) {
                    vector<unsigned>& lab = label[u];
                    vector<int>& pp = pos[u];
                    int insert_pos = pla1;
                    int delta = (int)cur_mod_labels[u].size();

                    while (pp.size() < dis) {
                        pp.push_back(lab.size());
                    }

                    lab.insert(lab.begin() + insert_pos, cur_mod_labels[u].begin(), cur_mod_labels[u].end());
                    // assert((lab[pos[u][dis - 1]] & MASK) == dis && (lab[insert_pos + delta - 1] & MASK) == dis);
                    // assert(pos[u][dis - 1] == 0 || ((lab[pos[u][dis - 1] - 1] & MASK) < dis));
                    // assert(insert_pos + delta == lab.size() || ((lab[insert_pos + delta] & MASK) > dis));
                    std::sort(lab.begin() + pos[u][dis - 1], lab.begin() + insert_pos + delta);
                    if ((size_t)dis < pp.size()) {
                        for (size_t ii = dis; ii < pp.size(); ++ii) pp[ii] += delta;
                    } else {
                        assert(pp.size() == dis);
                        pp.push_back((int)lab.size());
                    }
                }
            }
#pragma omp critical
            {
                cur_cnt += local_cnt;
                cand_size += local_cand_size;
                enum_size += local_enum_size;
                std::cout << "local prev = " << local_prev << ", local bad = " << local_bad << std::endl;
            }
        }
        double t2 = omp_get_wtime();
        total_new_cnt += cur_cnt;

#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            for (int u = pid; u < totalV; u += np) {
                prev_mod_labels[u] = std::move(cur_mod_labels[u]);
                cur_mod_labels[u] = std::vector<unsigned>();
            }
        }
        double t3 = omp_get_wtime();
        std::cout << "Distance: " << dis << "  Insert Cnt: " << cur_cnt << " Cand size = " << cand_size
                  << " Enum label cnt = " << enum_size << "time = " << t2 - t1 << " " << t3 - t2 << " " << t3 - t1
                  << endl;

        // #region agent log
        DBG_LOG("C", "IndexReorder:inserting", "end of insert dis", "\"data\":{\"dis\":%u,\"cur_cnt\":%lld}", dis,
                cur_cnt);
        // #endregion

        prev_cnt = cur_cnt;
        cur_cnt = 0;
    }
    double t4 = omp_get_wtime();
    std::cout << "insert = " << t4 - t3 << "s, total new cnt = " << total_new_cnt << std::endl;

    // #region agent log
    DBG_LOG("C", "IndexReorder:afterInsertLoop", "insert loop done, merging", "\"data\":{}");
    // #endregion

    // merge_labels(new_label);

    // double t5 = omp_get_wtime();
    // std::cout << "merge = " << t5 - t4 << "s" << std::endl;

    // #region agent log
    DBG_LOG("C", "IndexReorder:done", "IndexReorder fully complete", "\"data\":{}");
    // #endregion
}

// ============ for edge insertion =============

void Einsert(unsigned vid, vector<unsigned>& edges) {
    int pla = -1;
    for (int i = 0; i < edges.size(); ++i) {
        if (edges[i] > vid) {
            pla = i;
            break;
        }
    }

    if (pla == -1)
        edges.emplace_back(vid);
    else
        edges.emplace(edges.begin() + pla, vid);
}

void InsertGraph(float cc) {
    int cnt = 0;
    float c = 0.8;

    while (cnt < Dcnt) {
        int vid = (int)(totalV * cc) + cnt, adj = rand() % (int)(totalV * (1 - c)) + (int)(totalV * c);

        if (adj >= totalV) continue;

        auto it = find(con[vid].begin(), con[vid].end(), adj);

        if (it == con[vid].end()) {
            // cout<<vid<<"  "<<adj<<endl;

            Einsert(vid, conD[adj]);
            Einsert(adj, conD[vid]);

            Einsert(vid, con[adj]);
            Einsert(adj, con[vid]);
        }

        cnt += 1;
    }
}

void DynamicFile(int flg) {
    string name = "dynamic.edge";
    string s;
    const char* filepath = name.c_str();
    ifstream infile;

    infile.open(filepath);
    if (!infile.is_open()) {
        cout << "No such file!" << endl;
        exit(-1);
    }

    long xx = 0;

    while (getline(infile, s)) {
        ArrayOnHeap<char> strc(strlen(s.c_str()) + 1, ArrayOnHeap<char>::uninitialized);
        strcpy(strc.data(), s.c_str());
        char* s1 = strtok(strc.data(), " ");

        int ii = 0, va, vb;
        while (s1) {
            if (ii % 2 == 0)
                va = atoi(s1);
            else {
                vb = atoi(s1);
                DyEdges.push_back(make_pair(va, vb));
            }

            ii += 1;
            s1 = strtok(NULL, " ");
        }

        xx += 1;
    }

    for (int i = 0; i < DyEdges.size(); ++i) {
        int vid = DyEdges[i].first, adj = DyEdges[i].second;
        if (flg == 0) {
            Einsert(vid, conD[adj]);
            Einsert(adj, conD[vid]);

            Einsert(vid, con[adj]);
            Einsert(adj, con[vid]);
        } else {
            conD[vid].push_back(adj);  // for the index update
            conD[adj].push_back(vid);

            con[vid].erase(remove(con[vid].begin(), con[vid].end(), adj), con[vid].end());
            con[adj].erase(remove(con[adj].begin(), con[adj].end(), vid), con[adj].end());
        }
    }
}

int DisVerdict(unsigned w, unsigned dmax, vector<int>& nowDis) {
    // find a single path between tgt and w, where d<=dmax, return 1

    for (int i = 0; i < pos[w][dmax]; ++i) {
        unsigned vid = label[w][i] >> MAXMOV, dis = label[w][i] & MASK;

        if (nowDis[vid] == -1) continue;

        if (nowDis[vid] + dis <= dmax or nowDis[vid] == 0) return 1;
    }

    return -1;
}

int ifdelete(unsigned tgt, unsigned tdis, unsigned v, unsigned vdis) {  // remove tgt from label[v]
    int flg = -1;
    vector<int> vec(totalV, -1);

    for (int i = 0; i < pos[tgt][tdis - vdis]; ++i) vec[(label[tgt][i] >> MAXMOV)] = (label[tgt][i] & MASK);

    for (int i = 0; i < clab[tgt].size(); ++i) vec[clab[tgt][i]] = vdis;  // 可以起到覆盖作用

    for (int i = 0; i < clab[v].size(); ++i) {
        unsigned w = clab[v][i];

        flg = DisVerdict(w, tdis - vdis, vec);

        if (flg != -1) {
            flg = 1;
            break;
        }
    }

    return flg;
}

void Insert_Parallel() {
    omp_set_num_threads(threads);

    clab = ArrayOnHeap<vector<unsigned>>(totalV);
    cpos = ArrayOnHeap<vector<int>>(totalV);

    vaff.resize(totalV, -1);

    for (int u = 0; u < totalV; ++u) {
        cpos[u].push_back(0);

        for (int i = 0; i < conD[u].size(); ++i) {
            if (conD[u][i] < u) {
                unsigned elem = conD[u][i] << MAXMOV | 1;
                clab[u].push_back(elem);
            }
        }

        if (clab[u].size() > 0) {
            vector<unsigned>(clab[u]).swap(clab[u]);
            sort(clab[u].begin(), clab[u].end());
            vaff[u] = (clab[u][0] >> MAXMOV);
        }

        cpos[u].push_back(clab[u].size());
    }

    int dis = 2;
    for (long long cnt = 1, cnttt = 0; cnt > 0; ++dis) {
        cnt = 0, cnttt = 0;
        ArrayOnHeap<vector<unsigned>> label_new(totalV);

#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            long long local_cnt = 0;
            ArrayOnHeap<unsigned char> used(totalV / 8 + 1);
            vector<int> cand;

            ArrayOnHeap<char> nowdis(totalV, ArrayOnHeap<char>::uninitialized);
            nowdis.memset(-1);

            for (int u = pid; u < totalV; u += np) {
                cand.clear();

                // === Obtain new labels ===
                for (int i = 0; i < con[u].size(); ++i) {
                    int w = con[u][i];

                    for (int j = cpos[w][dis - 2]; j < cpos[w][dis - 1]; ++j) {
                        int v = clab[w][j] >> MAXMOV;

                        if (v >= u) break;

                        if (!(used[v / 8] & (1 << (v % 8)))) {
                            used[v / 8] |= (1 << (v % 8)), cand.push_back(v);
                        }
                    }
                }

                if (dis <= pos[0].size())
                    for (int i = 0; i < conD[u].size(); ++i) {
                        int w = conD[u][i];

                        for (int j = pos[w][dis - 2]; j < pos[w][dis - 1]; ++j) {
                            int v = label[w][j] >> MAXMOV;

                            if (v >= u) break;

                            if (!(used[v / 8] & (1 << (v % 8)))) {
                                used[v / 8] |= (1 << (v % 8)), cand.push_back(v);
                            }
                        }
                    }

                if (cand.size() == 0) continue;

                int n_cand = 0;
                int pla1 = dis < pos[0].size() ? pos[u][dis] : label[u].size();
                int cla1 = dis < cpos[0].size() ? cpos[u][dis] : clab[u].size();

                for (int i = 0; i < pla1; ++i) nowdis[label[u][i] >> MAXMOV] = label[u][i] & MASK;

                for (int i = 0; i < cla1; ++i) nowdis[clab[u][i] >> MAXMOV] = clab[u][i] & MASK;

                for (int i = 0; i < (int)cand.size(); ++i) {
                    used[cand[i] / 8] = 0;

                    if (nowdis[cand[i]] != -1) continue;  // 已经是直连，不需要再去判断

                    if (can_update_add(cand[i], dis, nowdis.data())) {
                        cand[n_cand++] = cand[i];
                    }
                }

                cand.resize(n_cand);
                sort(cand.begin(), cand.end());

                // if (cand.size() > 0){
                // cout<<" id: "<<u<<" :  ";
                // for (int ii=0; ii<cand.size(); ++ii){
                //     cout<<"$v_"<<cand[ii]<<"$ ";
                // }
                // cout<<endl;
                // }

                if (cand.size() > 0) {
                    if (vaff[u] == -1)
                        vaff[u] = (int)cand[0];
                    else
                        vaff[u] = min(vaff[u], (int)cand[0]);  // 后续剪枝
                }

                for (int i = 0; i < (int)cand.size(); ++i) {
                    label_new[u].push_back((((unsigned)cand[i]) << MAXMOV) | (unsigned)dis), ++local_cnt;
                }

                for (int i = 0; i < pla1; ++i) nowdis[label[u][i] >> MAXMOV] = -1;
                for (int i = 0; i < cla1; ++i) nowdis[clab[u][i] >> MAXMOV] = -1;
            }

#pragma omp critical
            {
                cnt += local_cnt;
            }
        }

#pragma omp parallel
        {
            // === insert clab to label ===
            int pid = omp_get_thread_num(), np = omp_get_num_threads();

            for (int u = pid; u < totalV; u += np) {
                clab[u].insert(clab[u].end(), label_new[u].begin(), label_new[u].end());
                vector<unsigned>(clab[u]).swap(clab[u]);
                vector<unsigned>().swap(label_new[u]);
                cpos[u].push_back(clab[u].size());
            }
        }
        cout << "dis: " << dis << "  cnt: " << cnt << endl;
    }

    // for (int i=0; i<totalV; ++i){
    //     cout<<vaff[i]<<endl;
    // }
}

int cand_remove(vector<int>& nowd, vector<unsigned>& lab, int dis) {
    for (int i = 0; i < lab.size(); ++i) {
        unsigned w = lab[i] >> MAXMOV, d = lab[i] & MASK;
        if (d >= dis) break;

        if (nowd[w] > -1 && nowd[w] + d <= dis) return 1;
    }

    return -1;
}

int cand_remove_no_prune(vector<int>& nowd, vector<unsigned>& lab, int dis) {
    for (int i = 0; i < lab.size(); ++i) {
        unsigned w = lab[i] >> MAXMOV, d = lab[i] & MASK;
        if (d >= dis) break;

        if (nowd[w] > -1 && nowd[w] + d < dis) return 1;
    }

    return -1;
}

// 针对的是 vaff[u] != -1 的顶点的判定
int cand_remove_2(vector<int>& nowD, int u, int v, int dis) {
    vector<unsigned>& lab = label[v];

    for (int d = 1; d < dis; ++d) {
        int dd = dis - d;  // clb 中的上限值
        unsigned vid = totalV, cnt = 0;
        dd = cpos[u].size() - 1 <= dd ? cpos[u].size() - 1 : dd;
        for (int j = 1; j <= dd; ++j) {
            if (cpos[u][j] == 0) continue;
            unsigned elem = clab[u][cpos[u][j] - 1] >> MAXMOV;
            vid = max(vid, elem);
            cnt += 1;
        }

        if (vid == totalV and cnt == 0) vid = 0;
        int pla1 = (d - 1 < pos[v].size()) ? pos[v][d - 1] : (int)lab.size();
        int pla2 = (d < pos[v].size()) ? pos[v][d] : (int)lab.size();

        for (int i = pla1; i < pla2; ++i) {
            unsigned w = lab[i] >> MAXMOV, dw = lab[i] & MASK;
            if (w > vid) break;  // 大于这个值表明，该距离值段的后续肯定没有交集
            if (nowD[w] > -1 && nowD[w] + dw <= dis) return 1;
        }
    }

    return -1;
}

void Insert_Remove_Parall() {
    vector<int> Rlab(totalV);
    long long cnt1 = 0, cnt2 = 0;
    Pla = ArrayOnHeap<vector<unsigned>>(totalV);

#pragma omp parallel
    {
        int pid = omp_get_thread_num(), np = omp_get_num_threads();
        vector<int> nowdis(totalV, -1), nowdis_extra(totalV, -1);
        long long c1 = 0, c2 = 0;

        for (int u = pid; u < totalV; u += np) {
            for (int i = 0; i < clab[u].size(); ++i) nowdis[clab[u][i] >> MAXMOV] = clab[u][i] & MASK;

            for (int i = 0; i < label[u].size(); ++i) {
                unsigned v = label[u][i] >> MAXMOV, d = label[u][i] & MASK;
                if (nowdis[v] != -1) {
                    Rlab[u] = 1, Pla[u].push_back(i);
                } else {
                    nowdis_extra[v] = d;
                }
            }

            // ==========================================

            for (int i = pos[u][0]; i < label[u].size(); ++i) {  // dis=0,1的label不需要check

                unsigned vid = label[u][i] >> MAXMOV, dis = label[u][i] & MASK;

                if (nowdis_extra[vid] == -1) continue;

                int flg = -1;

                if (vaff[u] == -1 and vaff[vid] != -1) {
                    flg = cand_remove(nowdis_extra, clab[vid], dis);
                }

                if (vaff[u] != -1 and vaff[vid] == -1 and vid > vaff[u]) flg = cand_remove_2(nowdis, u, vid, dis);

                if (vaff[u] != -1 and vaff[vid] != -1) {
                    flg = cand_remove(nowdis, clab[vid], dis);
                    if (flg == -1) flg = cand_remove(nowdis_extra, clab[vid], dis);
                    if (flg == -1) flg = cand_remove_2(nowdis, u, vid, dis);
                }

                if (flg == 1) Rlab[u] = 1, Pla[u].push_back(i);  // 对其他值的判定不会有影响
            }

            for (int i = 0; i < clab[u].size(); ++i) nowdis[clab[u][i] >> MAXMOV] = -1;

            for (int i = 0; i < label[u].size(); ++i) nowdis_extra[label[u][i] >> MAXMOV] = -1;
        }

        for (int u = pid; u < totalV; u += np) {
            vector<unsigned>& lab = label[u];

            if (Rlab[u] == 0) continue;

            sort(Pla[u].begin(), Pla[u].end());

            unsigned elem = (totalV - 1) << MAXMOV | 2;  // 理论上应该不是label值

            bool elemIsLabelValue = false;
            for (size_t i = 0; i < label[u].size(); ++i) {
                if (label[u][i] == elem) {
                    elemIsLabelValue = true;
                }
            }

            for (unsigned pla : Pla[u]) label[u][pla] = elem;

            int n_cand = 0;
            for (int i = 0; i < label[u].size(); ++i) {
                if (label[u][i] != elem) {
                    label[u][n_cand++] = label[u][i];
                }
            }
            label[u].resize(n_cand);
        }
    }
}

bool PosCheck(vector<unsigned>* lab, vector<int>* p) {
    bool ok = true;
    for (int u = 0; u < totalV; ++u) {
        // if (p[u].empty()) {
        //     cout << "pos[" << u << "] is empty!" << endl;
        //     ok = false;
        //     continue;
        // }

        // if (p[u][0] != 1) {
        //     cout << "pos[" << u << "][0] != 1, got " << p[u][0] << endl;
        //     ok = false;
        // }

        int prev = 0;
        for (int d = 0; d < (int)p[u].size(); ++d) {
            if (p[u][d] < prev || p[u][d] > (int)lab[u].size()) {
                cout << "pos[" << u << "][" << d << "] = " << p[u][d] << " out of range" << endl;
                ok = false;
                break;
            }
            for (int j = prev; j < p[u][d]; ++j) {
                unsigned actualDis = lab[u][j] & MASK;
                if (actualDis != (unsigned)d) {
                    cout << "label[" << u << "][" << j << "] dis=" << actualDis << " but expected " << d << endl;
                    ok = false;
                }
                if (d > 0 && j > prev) {
                    unsigned prevHub = lab[u][j - 1] >> MAXMOV;
                    unsigned curHub = lab[u][j] >> MAXMOV;
                    if (curHub < prevHub) {
                        cout << "label[" << u << "] not sorted at index " << j << " within dis=" << d << endl;
                        ok = false;
                    }
                }
            }
            prev = p[u][d];
        }
        for (size_t i = prev; i < lab[u].size(); ++i) {
            unsigned actualDis = lab[u][i] & MASK;
            if (actualDis != p[u].size()) {
                cout << "label[" << u << "][" << i << "] dis=" << actualDis << " but expected " << p[u].size() << endl;
                ok = false;
            }
        }
    }
    return ok;
}

bool LabelsEqual(const ArrayOnHeap<vector<unsigned>>* lhs, const ArrayOnHeap<vector<unsigned>>* rhs) {
    auto printLabelArray = [](const vector<unsigned>& arr, const char* name) {
        cout << name << " (size=" << arr.size() << "): ";
        for (size_t i = 0; i < arr.size(); ++i) cout << "(" << (arr[i] >> MAXMOV) << "," << (arr[i] & MASK) << ") ";
        cout << endl;
    };

    if (lhs == nullptr || rhs == nullptr) {
        cout << "[LabelsEqual] null pointer: lhs=" << (lhs == nullptr) << ", rhs=" << (rhs == nullptr) << endl;
        return false;
    }
    if (lhs->size() != rhs->size()) {
        cout << "[LabelsEqual] top-level size mismatch: lhs=" << lhs->size() << ", rhs=" << rhs->size() << endl;
        return false;
    }
    if ((int)lhs->size() != totalV || (int)rhs->size() != totalV) {
        cout << "[LabelsEqual] size != totalV: lhs=" << lhs->size() << ", rhs=" << rhs->size() << ", totalV=" << totalV
             << endl;
        return false;
    }

    for (int u = 0; u < totalV; ++u) {
        const vector<unsigned>& a = (*lhs)[u];
        const vector<unsigned>& b = (*rhs)[u];
        if (a.size() != b.size()) {
            cout << "[LabelsEqual] first mismatch at u=" << u << ": size differs, lhs=" << a.size()
                 << ", rhs=" << b.size() << endl;
            printLabelArray(a, "lhs label[u]");
            printLabelArray(b, "rhs label[u]");
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) {
                cout << "[LabelsEqual] first mismatch at u=" << u << ", i=" << i << ", lhs=" << a[i] << ", rhs=" << b[i]
                     << endl;
                printLabelArray(a, "lhs label[u]");
                printLabelArray(b, "rhs label[u]");
                return false;
            }
        }
    }
    return true;
}

bool check_label_validity(const char* stage) {
    if (label.empty() || pos.empty()) {
        cout << "[check_label_validity] " << stage << " skip: label/pos is empty" << endl;
        return false;
    }

    bool ok = true;
    int badCnt = 0;
    int maxReport = 20;
    for (int u = 0; u < totalV; ++u) {
        int prev = 0;
        for (int d = 0; d < (int)pos[u].size(); ++d) {
            int cur = pos[u][d];
            if (cur < prev || cur > (int)label[u].size()) {
                if (badCnt < maxReport) {
                    cout << "[check_label_validity] " << stage << " invalid: u=" << u << ", d=" << d
                         << ", prev=" << prev << ", cur=" << cur << ", labelSize=" << label[u].size()
                         << ", posSize=" << pos[u].size() << endl;
                }
                ++badCnt;
                ok = false;
            }
            prev = cur;
        }
    }

    if (ok) {
        cout << "[check_label_validity] " << stage << " PASS" << endl;
    } else {
        cout << "[check_label_validity] " << stage << " FAIL, total invalid entries = " << badCnt << endl;
    }

    for (int u = 0; u < totalV; ++u) {
        for (int i = 1; i < (int)label[u].size(); ++i) {
            unsigned prevDis = label[u][i - 1] & MASK;
            unsigned curDis = label[u][i] & MASK;
            if (curDis < prevDis) {
                cerr << "[check_label_validity] " << stage << " label distance not nondecreasing, u=" << u
                     << ", i=" << i << ", prevDis=" << prevDis << ", curDis=" << curDis << endl;
                exit(-1);
            }
        }
    }

    for (int u = 0; u < totalV; ++u) {
        std::vector<int> newPos;
        unsigned prevDis = 0;
        for (int k = 0; k < (int)label[u].size(); ++k) {
            unsigned d = label[u][k] & MASK;
            while (prevDis < d) {
                newPos.push_back(k);
                ++prevDis;
            }
        }
        newPos.push_back((int)label[u].size());
        // if (newPos != pos[u]) {
        //     cout << "[check_label_validity] mismatch at u=" << u << " newPos: ";
        //     for (int x : newPos) cout << x << " ";
        //     cout << " | pos[u]: ";
        //     for (int x : pos[u]) cout << x << " ";
        //     cout << endl;
        //     assert(0);
        // }
        pos[u] = std::move(newPos);
    }
    cout << "[check_label_validity] " << stage << " pos rebuilt from label" << endl;
    return ok;
}

void TestReorder() {
    cout << "=== TestReorder ===" << endl;

    GraphReorder();

    double t = omp_get_wtime();
    IndexReorder();
    cout << "Reorder time:  " << omp_get_wtime() - t << " s" << endl;

    ArrayOnHeap<vector<unsigned>> reorderLabel = std::move(label);
    ArrayOnHeap<vector<int>> reorderPos = std::move(pos);

    bool posOk = PosCheck(reorderLabel.data(), reorderPos.data());
    cout << "pos check: " << (posOk ? "PASS" : "FAIL") << endl;

    // 2. rebuild index from scratch and compare
    IndexBuild();

    bool labelOk = true;
    for (int u = 0; u < totalV; ++u) {
        if (label[u].size() != reorderLabel[u].size()) {
            cout << "label size mismatch at u=" << u << " build=" << label[u].size()
                 << " reorder=" << reorderLabel[u].size() << endl;

            cout << "  build   L[" << u << "] = ";
            for (int j = 0; j < (int)label[u].size(); ++j)
                cout << "(" << (label[u][j] >> MAXMOV) << "," << (label[u][j] & MASK) << ") ";
            cout << endl;

            cout << "  reorder L[" << u << "] = ";
            for (int j = 0; j < (int)reorderLabel[u].size(); ++j)
                cout << "(" << (reorderLabel[u][j] >> MAXMOV) << "," << (reorderLabel[u][j] & MASK) << ") ";
            cout << endl;

            labelOk = false;
            continue;
        }
        for (int j = 0; j < (int)label[u].size(); ++j) {
            if (label[u][j] != reorderLabel[u][j]) {
                cout << "label mismatch at u=" << u << " j=" << j << " build=(" << (label[u][j] >> MAXMOV) << ","
                     << (label[u][j] & MASK) << ")"
                     << " reorder=(" << (reorderLabel[u][j] >> MAXMOV) << "," << (reorderLabel[u][j] & MASK) << ")"
                     << endl;
                labelOk = false;
                break;
            }
        }
    }
    cout << "label check: " << (labelOk ? "PASS" : "FAIL") << endl;

    cout << "=== TestReorder Done ===" << endl;
}

unsigned Query(int u, int v) {
    if (u == v) return 0;
    unsigned lu = (unsigned)label[u].size(), lv = (unsigned)label[v].size(), dis = MAXD;
    for (int i = 0, j = 0; i < lu && j < lv; ++i) {
        for (; j < lv && label[v][j] >> MAXMOV < label[u][i] >> MAXMOV; ++j);
        if (j < lv && label[v][j] >> MAXMOV == label[u][i] >> MAXMOV)
            dis = min(dis, (label[u][i] & MASK) + (label[v][j] & MASK));
    }
    return dis;
}

void OrderBuild(string graphName) {
    string suffixes[] = {"d", "b", "s"};

    for (int si = 0; si < 3; ++si) {
        string orderFile = graphName + "_" + suffixes[si] + ".order";
        ifstream fin(orderFile);
        if (!fin.is_open()) {
            cout << "Cannot open order file: " << orderFile << ", skipping." << endl;
            continue;
        }

        con.clear();
        conD.clear();
        v2degree.clear();
        v2p.clear();
        p2v.clear();
        totalV = 0;
        label = ArrayOnHeap<vector<unsigned>>();
        pos = ArrayOnHeap<vector<int>>();

        GraphInitial(graphName + ".txt");

        vector<int> new_p2v(totalV);
        for (int i = 0; i < totalV; ++i) fin >> new_p2v[i];
        fin.close();

        GraphReorder_vp({}, new_p2v);

        cout << "=== Building index with order: " << suffixes[si] << " ===" << endl;
        double t = omp_get_wtime();
        IndexBuild();
        cout << "Build time (" << suffixes[si] << "): " << omp_get_wtime() - t << " s" << endl;

        IndexSize(0);

        string indexFile = graphName + "_" + suffixes[si] + ".bin";
        IndexSave(indexFile);
        cout << "Index saved to: " << indexFile << endl;
    }
}

void TestOrderReorder(string graphName, string srcOrder, string tgtOrder) {
    cout << "=== TestOrderReorder: " << srcOrder << " -> " << tgtOrder << " ===" << endl;

    // #region agent log
    DBG_LOG("E", "TestOrderReorder:entry", "starting test", "\"data\":{\"src\":\"%s\",\"tgt\":\"%s\"}",
            srcOrder.c_str(), tgtOrder.c_str());
    // #endregion

    con.clear();
    conD.clear();
    v2degree.clear();
    v2p.clear();
    p2v.clear();
    totalV = 0;
    label = ArrayOnHeap<vector<unsigned>>();
    pos = ArrayOnHeap<vector<int>>();

    GraphInitial(graphName + ".txt");

    {
        ifstream fin(graphName + "_" + srcOrder + ".order");
        if (!fin.is_open()) {
            cout << "Cannot open " << srcOrder << " order file!" << endl;
            return;
        }
        vector<int> src_p2v(totalV);
        for (int i = 0; i < totalV; ++i) fin >> src_p2v[i];
        GraphReorder_vp({}, src_p2v);
    }

    IndexLoad(graphName + "_" + srcOrder + ".bin");
    long long srcIndexLabelCnt = 0;
    for (int u = 0; u < totalV; ++u) srcIndexLabelCnt += (long long)label[u].size();
    cout << "src index label count = " << srcIndexLabelCnt << endl;

    // #region agent log
    DBG_LOG("E", "TestOrderReorder:afterSrcLoad", "src index loaded", "\"data\":{\"src\":\"%s\",\"tgt\":\"%s\"}",
            srcOrder.c_str(), tgtOrder.c_str());
    // #endregion

    {
        ifstream fin(graphName + "_" + tgtOrder + ".order");
        if (!fin.is_open()) {
            cout << "Cannot open " << tgtOrder << " order file!" << endl;
            return;
        }
        vector<int> tgt_p2v(totalV);
        for (int i = 0; i < totalV; ++i) fin >> tgt_p2v[i];

        double t = omp_get_wtime();
        GraphReorder_vp({}, tgt_p2v);
        cout << "Graph reorder time: " << omp_get_wtime() - t << " s" << endl;

        // #region agent log
        DBG_LOG("E", "TestOrderReorder:afterTgtReorder", "tgt graph+label reordered",
                "\"data\":{\"src\":\"%s\",\"tgt\":\"%s\"}", srcOrder.c_str(), tgtOrder.c_str());
        // #endregion

        t = omp_get_wtime();
        IndexReorder();
        cout << "Index reorder time: " << omp_get_wtime() - t << " s" << endl;
    }

    // #region agent log
    DBG_LOG("E", "TestOrderReorder:afterIndexReorder", "index reorder done", "\"data\":{\"src\":\"%s\",\"tgt\":\"%s\"}",
            srcOrder.c_str(), tgtOrder.c_str());
    // #endregion

    std::cout << "q = " << Query(24905, 162158) << std::endl;

    ArrayOnHeap<vector<unsigned>> reorderLabel = std::move(label);
    ArrayOnHeap<vector<int>> reorderPos = std::move(pos);

    IndexLoad(graphName + "_" + tgtOrder + ".bin");
    long long tgtIndexLabelCnt = 0;
    for (int u = 0; u < totalV; ++u) tgtIndexLabelCnt += (long long)label[u].size();
    cout << "tgt index label count = " << tgtIndexLabelCnt << endl;

    // #region agent log
    DBG_LOG("E", "TestOrderReorder:afterTgtLoad", "tgt index loaded for comparison",
            "\"data\":{\"src\":\"%s\",\"tgt\":\"%s\"}", srcOrder.c_str(), tgtOrder.c_str());
    // #endregion

    bool labelOk = true;
    int mismatchCnt = 0;
    for (int u = 0; u < totalV && mismatchCnt < 10; ++u) {
        if (label[u].size() != reorderLabel[u].size()) {
            cout << "label size mismatch at u=" << u << " target=" << label[u].size()
                 << " reorder=" << reorderLabel[u].size() << endl;

            cout << "  target  L[" << u << "] = ";
            for (int j = 0; j < (int)label[u].size(); ++j)
                cout << "(" << (label[u][j] >> MAXMOV) << "," << (label[u][j] & MASK) << ") ";
            cout << endl;

            cout << "  reorder L[" << u << "] = ";
            for (int j = 0; j < (int)reorderLabel[u].size(); ++j)
                cout << "(" << (reorderLabel[u][j] >> MAXMOV) << "," << (reorderLabel[u][j] & MASK) << ") ";
            cout << endl;

            labelOk = false;
            ++mismatchCnt;
            continue;
        }
        for (int j = 0; j < (int)label[u].size(); ++j) {
            if (label[u][j] != reorderLabel[u][j]) {
                cout << "label mismatch at u=" << u << " j=" << j << " target=(" << (label[u][j] >> MAXMOV) << ","
                     << (label[u][j] & MASK) << ")"
                     << " reorder=(" << (reorderLabel[u][j] >> MAXMOV) << "," << (reorderLabel[u][j] & MASK) << ")"
                     << endl;
                labelOk = false;
                ++mismatchCnt;
                break;
            }
        }
    }

    bool posOk = PosCheck(reorderLabel.data(), reorderPos.data());

    cout << "label check (" << srcOrder << " -> " << tgtOrder << "): " << (labelOk ? "PASS" : "FAIL") << endl;
    cout << "pos   check (" << srcOrder << " -> " << tgtOrder << "): " << (posOk ? "PASS" : "FAIL") << endl;

    cout << "=== TestOrderReorder Done ===" << endl;

    // #region agent log
    DBG_LOG("E", "TestOrderReorder:done", "test completed",
            "\"data\":{\"src\":\"%s\",\"tgt\":\"%s\",\"labelOk\":%d,\"posOk\":%d}", srcOrder.c_str(), tgtOrder.c_str(),
            (int)labelOk, (int)posOk);
    // #endregion
}

int main(int argc, char** argv) {
    threads = 1;
    // Graphpath = argv[1];
    // GraphInitial(argv[1]);
    // for (int i = 0; i < totalV; ++i) {
    //     assert(p2v[i] == i);
    //     assert(v2p[i] == i);
    // }
    // IndexBuild();
    // IndexPrint();
    // DeleteEdge(5, 6);
    // IndexDel_Parallel();
    // IndexDel_Add();
    // IndexSize(1);
    // IndexPrint();

    // return 0;
    if (argc != 8) {
        cout << "Please input 8 parameters" << endl;
        exit(-1);
    }

    Graphpath = argv[1];               // the path of graph
    Indexpath = argv[2];               // the path of 2-hop index
    program_choice = atoi(argv[3]);    // the execute program 0-5
    active_ratio = atof(argv[4]);      // the activated ratio of vertices
    threads = atoi(argv[5]);           // the number of threads
    dynamic_edge_num = atoi(argv[6]);  // the number of dynamic edges
    query_task_num = atoi(argv[7]);    // the number of query tasks

    GraphInitial(Graphpath);

    if (program_choice == 0) {
        float t = omp_get_wtime();

        IndexBuild();

        cout << "Build time:  " << omp_get_wtime() - t << " s" << endl;

        IndexSave(Indexpath);

    } else if (program_choice == 1) {
        cout << "Execute M2HL for edge deletion ......" << endl;

        IndexLoad(Indexpath);

        DeleteGraph(active_ratio);  // 必须先清理所有的删边，否则反向激活会有误差

        double t = omp_get_wtime();

        IndexDel_Parallel();

        IndexDel_Add();

        cout << "Update time:  " << omp_get_wtime() - t << " s" << endl;

        IndexSize(1);

    } else if (program_choice == 2) {
        cout << "Execute PSL for edge deletion ......" << endl;

        DeleteGraph(active_ratio);

        double t = omp_get_wtime();

        IndexBuild();

        cout << "Reconstruct time:  " << omp_get_wtime() - t << " s" << endl;

        IndexSize(0);

    } else if (program_choice == 3) {  // Edge insert

        cout << "Execute M2HL for edge insertion ......" << endl;

        IndexLoad(Indexpath);

        InsertGraph(active_ratio);

        double t = omp_get_wtime();

        Insert_Parallel();

        Insert_Remove_Parall();

        cout << "Update time:  " << omp_get_wtime() - t << " s" << endl;

        IndexSize(1);

    } else if (program_choice == 4) {
        cout << "Execute PSL for edge insertion ......" << endl;

        InsertGraph(active_ratio);

        double t = omp_get_wtime();

        IndexBuild();

        cout << "Reconstruct time:  " << omp_get_wtime() - t << " s" << endl;

        IndexSize(0);

    } else if (program_choice == 5) {
        IndexLoad(Indexpath);
        omp_set_num_threads(threads);
        printf("sorting...\n");
#pragma omp parallel
        {
            int pid = omp_get_thread_num(), np = omp_get_num_threads();
            for (int u = pid; u < totalV; u += np) sort(label[u].begin(), label[u].end());
        }
        double t = omp_get_wtime(), t1 = 0;
        for (int i = 0; i < query_task_num; ++i) {
            int u = rand() % totalV, v = rand() % totalV;
            int dis = Query(u, v);
        }
        t1 = omp_get_wtime() - t;
        printf("Average query time = %0.3lf ns\n", t1 * 1000000000.0 / query_task_num);
    } else if (program_choice == 6) {
        cout << "Test reorder after random edge deletion ......" << endl;

        IndexLoad(Indexpath);

        DeleteGraphRandByEdge(1000);

        double t = omp_get_wtime();

        IndexDel_Parallel();

        IndexDel_Add();

        cout << "Update time:  " << omp_get_wtime() - t << " s" << endl;

        TestReorder();
    } else if (program_choice == 7) {
        cout << "Build indexes from different p2v orders ......" << std::endl;
        OrderBuild(StripExt(Graphpath));
    } else if (program_choice == 8) {
        cout << "Test reordering between different p2v orders ......" << std::endl;
        std::string graphName = StripExt(Graphpath);
        TestOrderReorder(graphName, "d", "b");
        TestOrderReorder(graphName, "d", "s");
        TestOrderReorder(graphName, "b", "d");
        TestOrderReorder(graphName, "b", "s");
        TestOrderReorder(graphName, "s", "d");
        TestOrderReorder(graphName, "s", "b");
    } else if (program_choice == 9) {
        cout << "Test reorder after random edge insertion ......" << endl;

        IndexLoad(Indexpath);
        // check_label_validity("program9_after_index_load");

        // DeleteGraphRandByEdge(1000);
        InsertGraph(0.1);

        double t = omp_get_wtime();

        Insert_Parallel();
        double t1 = omp_get_wtime();

        Insert_Remove_Parall();
        double t2 = omp_get_wtime();

        merge_labels(clab);
        double t3 = omp_get_wtime();
        cout << "Update time:  " << t3 - t << " s, " << t1 - t << " " << t2 - t1 << " " << t3 - t2 << endl;

        check_label_validity("program9_after_insert_remove");
        ArrayOnHeap<std::vector<unsigned>> updateLabel = std::move(label);
        ArrayOnHeap<std::vector<int>> updatePos = std::move(pos);
        IndexBuild();
        check_label_validity("program9_after_index_build");
        bool equal = LabelsEqual(&updateLabel, &label);
        assert(equal);

        TestReorder();
    }

    return 0;
}
