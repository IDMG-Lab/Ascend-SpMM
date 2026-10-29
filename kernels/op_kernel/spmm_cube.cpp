// #define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

template<typename T1, typename T2> class KernelMmad {
public:
    static constexpr size_t NUM_DATABLOCK_ELES = 32 / sizeof(T2);
    static constexpr size_t ROW_WINDOW = 16;
    static constexpr size_t LOG_ROW_WINDOW = 4;
    static constexpr size_t CUBE_BLOCK = 16;
    __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, GM_ADDR workspace, size_t size, size_t len, size_t D, TPipe* pipeIn)
    {
        // set cube only

        // aGM.SetGlobalBuffer((__gm__ T2 *)a);
        // bGM.SetGlobalBuffer((__gm__ T2 *)b);
        // cGM.SetGlobalBuffer((__gm__ float *)c);
        m = ROW_WINDOW; k = NUM_DATABLOCK_ELES; n = D;
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        pipe = pipeIn;
        pipe->InitBuffer(inQueueA1, 1, aSize * sizeof(T2));
        pipe->InitBuffer(inQueuetemp, 1, aSize * sizeof(T2));
        pipe->InitBuffer(inQueueA2, 1, aSize * sizeof(T2));
        pipe->InitBuffer(inQueueB1, 1, bSize * sizeof(T2));
        pipe->InitBuffer(inQueueB2, 1, bSize * sizeof(T2));
        pipe->InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        
        rowptrGlobal.SetGlobalBuffer((__gm__ T1*)rowptr);
        colGlobal.SetGlobalBuffer((__gm__ T1*)col);
        startGlobal.SetGlobalBuffer((__gm__ T1*)start);
        valueGlobal.SetGlobalBuffer((__gm__ T2*)value);
        denseGlobal.SetGlobalBuffer((__gm__ T2*)dense);
        yGlobal.SetGlobalBuffer((__gm__ T2*)y);
        workGlobal.SetGlobalBuffer((__gm__ T2*)workspace + (GetBlockIdx() * ROW_WINDOW * NUM_DATABLOCK_ELES), ROW_WINDOW * NUM_DATABLOCK_ELES);

        size_t L = GetBlockIdx() * len, R = (GetBlockIdx() + 1) * len;
        if (L > size) L = size;
        if (R > size) R = size;
        this->L = L; this->R = R; this->D = D;
        this->rowptr_idx = startGlobal.GetValue(GetBlockIdx());
        this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
    }
    __aicore__ inline void Process()
    {
        size_t src_offset, dst_offset = this->rowptr_idx * this->D * ROW_WINDOW, d;
        AscendC::LocalTensor<T2> b1Local = inQueueB1.AllocTensor<T2>();
        ZERO = inQueuetemp.AllocTensor<T2>();
        AscendC::InitConstValue(ZERO, {1, static_cast<uint16_t>(ROW_WINDOW * NUM_DATABLOCK_ELES * sizeof(T2) / 32), 0, (T2)0});
        AscendC::DataCopy(workGlobal, ZERO, 16 * NUM_DATABLOCK_ELES);
        int last_col = -1, cnt = -1;
        for (size_t i = L; i < R; i++)
        {
            while (i == this->next_rowptr)
            {
                if (cnt >= 0)
                {
                    AscendC::LocalTensor<T2> a1Local = inQueueA1.AllocTensor<T2>();
                    AscendC::DataCacheCleanAndInvalid<T2, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(workGlobal);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
                    AscendC::DataCopy(a1Local, workGlobal, 16 * NUM_DATABLOCK_ELES);
                    inQueueA1.EnQue(a1Local);
                    inQueueB1.EnQue(b1Local);
                    SplitA(dst_offset);
                    SplitB(dst_offset);
                    b1Local = inQueueB1.AllocTensor<T2>();
                    Compute();
                    CopyOut(dst_offset);
                    last_col = cnt = -1;
                }
                this->rowptr_idx++;
                dst_offset += this->D * ROW_WINDOW;
                this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
            }
            T1 col = colGlobal.GetValue(i), row_id = col & (ROW_WINDOW - 1);
            col >>= LOG_ROW_WINDOW;
            T2 v = valueGlobal.GetValue(i);
            src_offset = col * this->D;
            // if (dst_offset == 114*this->D) printf("%d %d %d %d %f\n",GetBlockIdx(), col, this->next_rowptr, i, v);
            if (last_col != col)
            {
                cnt ++;
                if (cnt == NUM_DATABLOCK_ELES)
                {
                    cnt = 0;
                    AscendC::LocalTensor<T2> a1Local = inQueueA1.AllocTensor<T2>();
                    AscendC::DataCacheCleanAndInvalid<T2, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(workGlobal);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
                    AscendC::DataCopy(a1Local, workGlobal, 16 * NUM_DATABLOCK_ELES);
                    inQueueA1.EnQue(a1Local);
                    inQueueB1.EnQue(b1Local);
                    SplitA(dst_offset);
                    SplitB(dst_offset);
                    b1Local = inQueueB1.AllocTensor<T2>();
                    Compute();
                    CopyOut(dst_offset);
                }
                last_col = col;
                AscendC::DataCopyParams copyParams;
                if constexpr(sizeof(T2) == 2)
                {
                    copyParams.blockCount = this->D / 16;
                    copyParams.blockLen = 1; // 搬运的单位为DataBlock(32Byte)，每个DataBlock内有8个float
                    copyParams.srcStride = 0; // 表示两次搬运src之间的间隔，单位为DataBlock
                    copyParams.dstStride = 15;
                }
                if constexpr(sizeof(T2) == 4)
                {
                    copyParams.blockCount = this->D / 8;
                    copyParams.blockLen = 1; // 搬运的单位为DataBlock(32Byte)，每个DataBlock内有8个float
                    copyParams.srcStride = 0; // 表示两次搬运src之间的间隔，单位为DataBlock
                    copyParams.dstStride = 15;
                }
                AscendC::DataCopy(b1Local[cnt * 16], denseGlobal[src_offset], copyParams);
            }
            int pos = row_id * NUM_DATABLOCK_ELES + cnt;
            workGlobal.SetValue(pos, v);
            // if (dst_offset==4 * this->D * ROW_WINDOW)
            // {
            //     printf("%d %d %d %d %f %f %f\n",GetBlockIdx(), col, this->next_rowptr, i, v, denseGlobal.GetValue(src_offset), yGlobal.GetValue(0));
            // }
        }
        if (cnt >= 0)
        {
            AscendC::LocalTensor<T2> a1Local = inQueueA1.AllocTensor<T2>();
            AscendC::DataCacheCleanAndInvalid<T2, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(workGlobal);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
            AscendC::DataCopy(a1Local, workGlobal, 16 * NUM_DATABLOCK_ELES);
            inQueueA1.EnQue(a1Local);
            inQueueB1.EnQue(b1Local);
            SplitA(dst_offset);
            SplitB(dst_offset);
            Compute();
            CopyOut(dst_offset);
            last_col = cnt = -1;
        }
    }

private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len) {
        return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
    }

    __aicore__ inline void SplitA(size_t dst_offset)
    {
        AscendC::LocalTensor<T2> a1Local = inQueueA1.DeQue<T2>();
        AscendC::LocalTensor<T2> a2Local = inQueueA2.AllocTensor<T2>();
        AscendC::DataCopy(workGlobal, ZERO, 16 * NUM_DATABLOCK_ELES);

        // if (dst_offset==4 * this->D * ROW_WINDOW)
        // {
        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(NUM_DATABLOCK_ELES)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     DumpTensor(a1Local, 0, NUM_DATABLOCK_ELES * 16, shapeInfo);
        // }
        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(m);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadDataParams);

        inQueueA2.EnQue<T2>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(size_t dst_offset)
    {
        AscendC::LocalTensor<T2> b1Local = inQueueB1.DeQue<T2>();
        AscendC::LocalTensor<T2> b2Local = inQueueB2.AllocTensor<T2>();

        // if (dst_offset==4 * this->D * ROW_WINDOW)
        // {
        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(n)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     DumpTensor(b1Local, 0, n * 16, shapeInfo);
        // }
        // Nz -> Zn
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(n);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        if constexpr(sizeof(T2) == 4) loadDataParams.dstFracGap = loadDataParams.repeatTimes - 1;
        AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);
        
        AscendC::InitConstValue(ZERO, {1, static_cast<uint16_t>(NUM_DATABLOCK_ELES * n * sizeof(T2) / 32), 0, (T2)0});

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<T2>(b2Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T2> a2Local = inQueueA2.DeQue<T2>();
        AscendC::LocalTensor<T2> b2Local = inQueueB2.DeQue<T2>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut(size_t dst_offset)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr(sizeof(T2) == 2) fixpipeParams.quantPre = QuantMode_t::F322F16;
        // if (dst_offset==4 * this->D * ROW_WINDOW)
        // {
        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(n)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     DumpTensor(c1Local, 0, n * 16, shapeInfo);
        //     DumpTensor(yGlobal[dst_offset], 0, n * 16, shapeInfo);
        // }
        AscendC::SetAtomicAdd<T2>();
        AscendC::Fixpipe(yGlobal[dst_offset], c1Local, fixpipeParams);
        AscendC::SetAtomicNone();
        outQueueCO1.FreeTensor(c1Local);
        // if (dst_offset==4 * this->D * ROW_WINDOW)
        // {
        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(n)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     DumpTensor(c1Local, 0, n * 16, shapeInfo);
        //     DumpTensor(yGlobal[dst_offset], 0, n * 16, shapeInfo);
        // }
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueuetemp;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    size_t L, R, D, rowptr_idx, next_rowptr;
    AscendC::GlobalTensor<T1> rowptrGlobal, colGlobal, startGlobal;
    AscendC::GlobalTensor<T2> valueGlobal, denseGlobal, yGlobal, workGlobal;
    AscendC::LocalTensor<T2> ZERO;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
};
extern "C"  __global__ __aicore__ void spmm_cube(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR tempspace, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // TODO: user kernel impl
    // if ASCEND_IS_AIC {
        TPipe pipe;
        KernelMmad<DTYPE_COL, DTYPE_DENSE> op;
        op.Init(rowptr, col, value, dense, start, y, workspace, tiling_data.size, tiling_data.len, tiling_data.D, &pipe);
        op.Process();
    // }
    // if ASCEND_IS_AIV {
    //     return;
    // }
}