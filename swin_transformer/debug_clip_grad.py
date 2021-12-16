import time
import numpy as np
import oneflow as flow
# import torch as flow

def run():
    arr = np.random.randn(128, 30, 224, 224)    
    input = flow.tensor(arr, dtype=flow.float32, device="cuda", requires_grad=True)
    start = time.time()
    for idx in range(1000):
        m = flow.nn.ReLU()
        out = m(input)
        out = out.sum()
        out.backward()
        grad_norm = flow.nn.utils.clip_grad_norm_(input, 5.0)
        print("iter >>>>>>>> ", idx+1, " grad_norm >>>>>>> ", grad_norm.cpu().numpy().sum())
    print("cost: ", time.time()-start)


if __name__ == '__main__':
    # run without profile >>> bash debug_with_real_data.sh
    # run()

    # run with line_profiler profile >>> bash debug_clip_grad.sh > line_profile_clipgrad_flow.log 2>&1
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(run)
    # lp_wrapper()
    # lp.print_stats()

    # run with cProfile profile >>> bash debug_clip_grad.sh > cProfile_clipgrad_flow.log 2>&1
    import cProfile, pstats
    cp = cProfile.Profile()
    cp.enable()
    run()
    cp.disable()
    stats = pstats.Stats(cp).sort_stats('cumtime')
    stats.print_stats()





