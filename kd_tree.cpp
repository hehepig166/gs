#include <bits/stdc++.h>
using namespace std;


const int MAX_DIM = 1024;
const int MAX_N = 50000;
const float INF = std::numeric_limits<float>::infinity();


int n, dim, k;
priority_queue<pair<float, int>> Q;

int cmp_cnt;

//===============================================
//===============================================

inline float sqr(float x) {return x*x;}

struct Point {
    float coords[MAX_DIM];

    bool input() {
        for (int i=0; i<dim; i++) if (!scanf("%f", &coords[i])) return false;
        return true;
    }

    float sqrdis_to(const Point &p) {
        cmp_cnt++;
        float ret = 0;
        for (int i=0; i<dim; i++) ret += sqr(coords[i]-p.coords[i]);
        return ret;
    }

    float& operator[](int id) {return coords[id];}
    const float operator[](int id) const {return coords[id];}

} points[MAX_N*2+10];

struct Area {
    float mn[MAX_DIM];
    float mx[MAX_DIM];

    float mindis_to(const Point &x) {
        float ret = 0;
        for (int i=0; i<dim; i++) {
            ret += sqr(max(.0f,mn[i]-x.coords[i])+max(.0f, x.coords[i]-mx[i]));
        }
        return ret;
    }
};

struct Node {
    Node *lson, *rson;
    int p;                  // mid point id
    Area area;              // area
    int total;              // cnt of point in the area

    Node(): lson(NULL), rson(NULL), p(0), total(0) {}

    void update() {
        for (int i=0; i<dim; i++) {
            area.mn[i] = area.mx[i] = points[p][i];
            if (lson) {
                area.mn[i] = min(area.mn[i], lson->area.mn[i]);
                area.mx[i] = max(area.mx[i], lson->area.mx[i]);
            }
            if (rson) {
                area.mn[i] = min(area.mn[i], rson->area.mn[i]);
                area.mx[i] = max(area.mx[i], rson->area.mx[i]);
            }
            total = (lson ? lson->total : 0) + (rson ? rson->total : 0);
        }
    }
} memory[MAX_N*2+10];


Node *mem;
Node *root;
int pids[MAX_N*2+10];

class cmpbyaxis {
public:
    explicit cmpbyaxis(const int &_axis, const Point _ps[]): axis(_axis), ps(_ps) {}
    bool operator()(int x, int y) {return ps[x][axis] < ps[y][axis];}
private:
    const int axis;
    const Point *ps;
};



//===============================================
//===============================================

Point ask;

void build(Node *&cur, const int L, const int R, const int axis) {
    int mid = (L+R)>>1;
    cur = mem++;

    nth_element(pids+L, pids+mid, pids+R, cmpbyaxis(axis, points));

    int new_axis = (axis+1 >= dim ? 0 : axis+1);

    if (L < mid) build(cur->lson, L, mid-1, new_axis);
    if (R > mid) build(cur->rson, mid+1, R, new_axis);

    cur->p = pids[mid];
    cur->update();
}



void query(Node *cur) {
    if (!cur) return;

    float dist = ask.sqrdis_to(points[cur->p]);
    float l_mindis = (cur->lson ? cur->lson->area.mindis_to(ask) : INF);
    float r_mindis = (cur->rson ? cur->rson->area.mindis_to(ask) : INF);

    if (dist < Q.top().first) {
        Q.pop();
        Q.push({dist, cur->p});
    }

    if (l_mindis < r_mindis) {
        if (l_mindis <= Q.top().first) query(cur->lson);
        if (r_mindis <= Q.top().first) query(cur->rson);
    }
    else {
        if (r_mindis <= Q.top().first) query(cur->rson);
        if (l_mindis <= Q.top().first) query(cur->lson);
    }
}


void dfs_show(Node *cur) {
    printf("[%p]\n", cur);
    printf("  lrson:  %p %p\n", cur->lson, cur->rson);
    printf("  pid:    %d (%.03f", cur->p, points[cur->p][0]); for (int i=1; i<dim; i++) printf(", %.03f", points[cur->p][i]); puts(")");
    printf("  total:  %d\n", cur->total);
    printf("  area:  ");
    for (int i=0; i<dim; i++) printf(" [%.03f, %.03f]", cur->area.mn[i], cur->area.mx[i]);
    puts("");

    if (cur->lson) dfs_show(cur->lson);
    if (cur->rson) dfs_show(cur->rson);
}

//===============================================
//===============================================

void init() {
    scanf("%d%d", &n, &dim);
    for (int i=0; i<n; i++) {
        points[i].input();
    }

    mem = memory;
    root = NULL;
    for (int i=0; i<n; i++) pids[i] = i;
    build(root, 0, n-1, 0);
}

void solve_kdtree() {
    for (int i=0; i<k; i++) Q.push({INF, -1});

    cmp_cnt = 0;

    query(root);

    //printf("[%d] ", cmp_cnt);

    for (; Q.size(); Q.pop()) {
        if (Q.top().second < 0) continue;
        printf("%d ", Q.top().second);
    }
    puts("");
}

void debug() {
    puts("================================");
    for (int i=0; i<n; i++) {
        printf("[%03d] (%.03f", i, points[i][0]);
        for (int j=1; j<dim; j++) printf(", %.03f", points[i][j]);
        puts(")");
    }
    puts("================================");
    dfs_show(root);
    puts("================================");
}

//===============================================
//===============================================

int main()
{
    
    
    init();
    puts("ok");
    fflush(stdout);

    // debug();

    clock_t start, end;
    start = clock();

    scanf("%d", &k);
    while (ask.input()) {
        solve_kdtree();
        fflush(stdout);
    }

    end = clock();
    double dur = (double)(end-start)/CLOCKS_PER_SEC;
    fprintf(stderr, "use time: %.05f seconds\n", dur);

    return 0;
}