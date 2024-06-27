#include "all.h"

int main()
{
    auto params = read_params();
    auto scene = read_scene();
    auto orig_gs = init_gs(scene);

    int max_iter = 30000;
    for (int iter = 1; iter <= max_iter; iter++) {
        auto render_res = render();
        auto gt_res = get_gt();
    }

}