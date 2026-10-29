#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include<vector>

using namespace AscendC;
#define SPTILE 96
#define BLOCK 64
#define MAXL 1024

constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

static constexpr size_t COL_TILE = 16;
static constexpr size_t ROW_WINDOW = 16;
static constexpr size_t LOG_ROW_WINDOW = 4;
static constexpr size_t CUBE_BLOCK = 16;
template<typename T1, typename T2> class KernelMmad {
public:
    static_assert(sizeof(T2) == 2 || sizeof(T2) == 4, "T2 must be 2 or 4 bytes");
    static constexpr size_t MAXD = sizeof(T2) == 2 ? 1024 : 512;
    static constexpr size_t NUM_DATABLOCK_ELES = 32 / sizeof(T2);
    __aicore__ inline void Init(GM_ADDR dense, GM_ADDR y, GM_ADDR value, GM_ADDR start, GM_ADDR rowptr, GM_ADDR col_id, size_t size, size_t len, size_t D, TPipe* pipeIn)
    {
        // aGM.SetGlobalBuffer((__gm__ T2 *)a);
        // bGM.SetGlobalBuffer((__gm__ T2 *)b);
        // cGM.SetGlobalBuffer((__gm__ float *)c);
        pipe = pipeIn;
        pipe->InitBuffer(inQueueA1, 2, ROW_WINDOW * SPTILE * sizeof(T2));
        pipe->InitBuffer(inQueueA2, 2, ROW_WINDOW * SPTILE * sizeof(T2));
        pipe->InitBuffer(inQueueB1, 2, SPTILE * MAXD * sizeof(T2));
        pipe->InitBuffer(inQueueB2, 2, COL_TILE * MAXD * sizeof(T2));
        pipe->InitBuffer(outQueueCO1, 2, ROW_WINDOW * MAXD * sizeof(float));
        
        denseGlobal.SetGlobalBuffer((__gm__ T2*)dense);
        yGlobal.SetGlobalBuffer((__gm__ T2*)y);
        valueGlobal.SetGlobalBuffer((__gm__ T2*)value);
        colGlobal.SetGlobalBuffer((__gm__ T1*)col_id);
        rowptrGlobal.SetGlobalBuffer((__gm__ T1*)rowptr);
        startGlobal.SetGlobalBuffer((__gm__ T1*)start);

        rowptrGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // colGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        startGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // valueGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        size_t L = GetBlockIdx() * len, R = (GetBlockIdx() + 1) * len;
        if (L > size) L = size;
        if (R > size) R = size;
        this->L = L; this->R = R; this->D = D;
        this->rowptr_idx = L >= R ? 0 : startGlobal.GetValue(GetBlockIdx());
        this->next_rowptr = L >= R ? 0 : rowptrGlobal.GetValue(this->rowptr_idx + 1);
    }
    __aicore__ inline void Process()
    {
        size_t src_offset, dst_offset = this->rowptr_idx * this->D * ROW_WINDOW, d;
        // AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(0x8);
        
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
        AscendC::MmadParams mmadParams;
        mmadParams.m = ROW_WINDOW;
        mmadParams.k = COL_TILE;
                        mmadParams.n = D;
        while (L < R)
        {
        // AscendC::SyncAll<false>();
            // printf("%d %d\n",GetBlockIdx(), this->rowptr_idx);
            
            size_t r = min(this->next_rowptr, R), SpStart;
            if (L < r)
            {
                AscendC::LocalTensor<T2> a1Local, a2Local, b1Local, b2Local;
                AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
                mmadParams.cmatrixInitVal = true;
                // printf("%d %d\n", L, r);
                #pragma unroll
                for (SpStart = L; SpStart + SPTILE <= r ; SpStart += SPTILE)
                {
                    CopyInA(SpStart);
                    CopyInB1(SpStart);
                    a2Local = inQueueA2.DeQue<T2>();
                    b1Local = inQueueB1.DeQue<T2>();
                    #pragma unroll
                    for (size_t i = 0; i < SPTILE / COL_TILE; i++)
                    {
                        b2Local = inQueueB2.AllocTensor<T2>();
                        AscendC::LoadData2dTransposeParams loadDataParams;
                        loadDataParams.repeatTimes = D / COL_TILE;
                        loadDataParams.srcStride = 1;
                        if constexpr(sizeof(T2) == 4) loadDataParams.dstFracGap = loadDataParams.repeatTimes - 1;
                        // if constexpr(sizeof(T2) == 2) loadDataParams.dstGap = 0, loadDataParams.dstFracGap = 0;
                        AscendC::LoadDataWithTranspose(b2Local, b1Local[i * COL_TILE * MAXD], loadDataParams);
                        inQueueB2.EnQue<T2>(b2Local);
                        b2Local = inQueueB2.DeQue<T2>();
                        AscendC::Mmad(c1Local, a2Local[i * ROW_WINDOW * COL_TILE], b2Local, mmadParams);
                        PipeBarrier<PIPE_M>();
                        // if (GetBlockIdx()==0)
                        // {
                        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(16)};
                        //     AscendC::ShapeInfo shapeInfo(2, array);
                        //     DumpTensor(c1Local, 0, 16 * 16, shapeInfo);
                        // }
                        mmadParams.cmatrixInitVal = false;
                        inQueueB2.FreeTensor(b2Local);
                    }
                    inQueueA2.FreeTensor(a2Local);
                    inQueueB1.FreeTensor(b1Local);
                }
                size_t rem_tile = r - SpStart;
                if (rem_tile)
                {
                    CopyInAlast(SpStart, rem_tile);
                    CopyInB1last(SpStart, rem_tile);
                    a2Local = inQueueA2.DeQue<T2>();
                    b1Local = inQueueB1.DeQue<T2>();
                    size_t i;
                    #pragma unroll
                    for (i = 0; i < (rem_tile - 1) / COL_TILE + 1; i++)
                    {
                        b2Local = inQueueB2.AllocTensor<T2>();
                        AscendC::LoadData2dTransposeParams loadDataParams;
                        loadDataParams.repeatTimes = D / COL_TILE;
                        loadDataParams.srcStride = 1;
                        if constexpr(sizeof(T2) == 4) loadDataParams.dstFracGap = loadDataParams.repeatTimes - 1;
                        // if constexpr(sizeof(T2) == 2) loadDataParams.dstGap = 0, loadDataParams.dstFracGap = 0;
                        AscendC::LoadDataWithTranspose(b2Local, b1Local[i * COL_TILE * MAXD], loadDataParams);
                        inQueueB2.EnQue<T2>(b2Local);
                        b2Local = inQueueB2.DeQue<T2>();
                        AscendC::Mmad(c1Local, a2Local[i * ROW_WINDOW * COL_TILE], b2Local, mmadParams);
                        PipeBarrier<PIPE_M>();
                        
                        // if (GetBlockIdx()==0)
                        // {
                        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(16)};
                        //     AscendC::ShapeInfo shapeInfo(2, array);
                        //     DumpTensor(c1Local, 0, 16 * 16, shapeInfo);
                        // }

                        mmadParams.cmatrixInitVal = false;
                        inQueueB2.FreeTensor(b2Local);
                    }
                    inQueueA2.FreeTensor(a2Local);
                    inQueueB1.FreeTensor(b1Local);
                }
                
                outQueueCO1.EnQue<float>(c1Local);
                CopyOut(dst_offset);
                L = r;
            }
            this->rowptr_idx++;
            dst_offset += this->D * ROW_WINDOW;
            this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
        }
    }

private:
    __aicore__ inline void CopyInA(size_t SpStart)
    {
        AscendC::LocalTensor<T2> a1Local = inQueueA1.AllocTensor<T2>();
        if constexpr(sizeof(T2) == 2) AscendC::DataCopy(a1Local, valueGlobal[SpStart * ROW_WINDOW], ROW_WINDOW * SPTILE);
        if constexpr(sizeof(T2) == 4)
        {
            // AscendC::Nd2NzParams intriParamsIn{1, COL_TILE, ROW_WINDOW, 0, ROW_WINDOW, COL_TILE, 1, 0};
            // AscendC::DataCopy(a1Local, valueGlobal[SpStart * ROW_WINDOW], intriParamsIn);
            // AscendC::DataCopy(a1Local[1 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 1 * COL_TILE * ROW_WINDOW], intriParamsIn);
            // AscendC::DataCopy(a1Local[2 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 2 * COL_TILE * ROW_WINDOW], intriParamsIn);
            // AscendC::DataCopy(a1Local[3 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 3 * COL_TILE * ROW_WINDOW], intriParamsIn);
            // AscendC::DataCopy(a1Local[4 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 4 * COL_TILE * ROW_WINDOW], intriParamsIn);
            // AscendC::DataCopy(a1Local[5 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 5 * COL_TILE * ROW_WINDOW], intriParamsIn);
            AscendC::Nd2NzParams intriParamsIn{SPTILE / COL_TILE, COL_TILE, ROW_WINDOW, COL_TILE * ROW_WINDOW, ROW_WINDOW, COL_TILE, 1, COL_TILE * ROW_WINDOW};
            AscendC::DataCopy(a1Local, valueGlobal[SpStart * ROW_WINDOW], intriParamsIn);
        }
        inQueueA1.EnQue(a1Local);
        
        a1Local = inQueueA1.DeQue<T2>();
        AscendC::LocalTensor<T2> a2Local = inQueueA2.AllocTensor<T2>();
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.repeatTimes = SPTILE / COL_TILE;
        loadDataParams.srcStride = 1;
        if constexpr(sizeof(T2) == 4) loadDataParams.dstGap = 1;
        AscendC::LoadDataWithTranspose(a2Local, a1Local, loadDataParams);
        inQueueA2.EnQue<T2>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void CopyInAlast(size_t SpStart, uint16_t len)
    {
        AscendC::LocalTensor<T2> a1Local = inQueueA1.AllocTensor<T2>();
        AscendC::InitConstValue(a1Local, {1, static_cast<uint16_t>(ROW_WINDOW * SPTILE * sizeof(T2) / 32), 0, 0});
        if constexpr(sizeof(T2) == 2) AscendC::DataCopy(a1Local, valueGlobal[SpStart * ROW_WINDOW], ROW_WINDOW * len);
        if constexpr(sizeof(T2) == 4)
        {
            uint16_t cnt = len / COL_TILE, rem = len % COL_TILE;
            // AscendC::Nd2NzParams intriParamsIn{1, COL_TILE, ROW_WINDOW, 0, ROW_WINDOW, COL_TILE, 1, 0};
            // switch (cnt)
            // {
            //     case 5: AscendC::DataCopy(a1Local[4 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 4 * COL_TILE * ROW_WINDOW], intriParamsIn); [[fallthrough]];
            //     case 4: AscendC::DataCopy(a1Local[3 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 3 * COL_TILE * ROW_WINDOW], intriParamsIn); [[fallthrough]];
            //     case 3: AscendC::DataCopy(a1Local[2 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 2 * COL_TILE * ROW_WINDOW], intriParamsIn); [[fallthrough]];
            //     case 2: AscendC::DataCopy(a1Local[1 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 1 * COL_TILE * ROW_WINDOW], intriParamsIn); [[fallthrough]];
            //     case 1: AscendC::DataCopy(a1Local[0 * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + 0 * COL_TILE * ROW_WINDOW], intriParamsIn); [[fallthrough]];
            //     default: break;
            // }
            // if (rem) AscendC::DataCopy(a1Local[cnt * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + cnt * COL_TILE * ROW_WINDOW], {1, rem, ROW_WINDOW, 0, ROW_WINDOW, COL_TILE, 1, 0});
            
            AscendC::Nd2NzParams intriParamsIn{cnt, COL_TILE, ROW_WINDOW, COL_TILE * ROW_WINDOW, ROW_WINDOW, COL_TILE, 1, COL_TILE * ROW_WINDOW};
            if (cnt) AscendC::DataCopy(a1Local, valueGlobal[SpStart * ROW_WINDOW], intriParamsIn);
            if (rem) AscendC::DataCopy(a1Local[cnt * COL_TILE * ROW_WINDOW], valueGlobal[SpStart * ROW_WINDOW + cnt * COL_TILE * ROW_WINDOW], {1, rem, ROW_WINDOW, 0, ROW_WINDOW, COL_TILE, 1, 0});
        }
        inQueueA1.EnQue(a1Local);
        // if (GetBlockIdx()==0)
        // {
        //     uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(64)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     DumpTensor(a1Local, 0, 16 * 64, shapeInfo);
        // }
        
        a1Local = inQueueA1.DeQue<T2>();
        AscendC::LocalTensor<T2> a2Local = inQueueA2.AllocTensor<T2>();
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.repeatTimes = (len - 1) / COL_TILE + 1;
        loadDataParams.srcStride = 1;
        if constexpr(sizeof(T2) == 4) loadDataParams.dstGap = 1;
        AscendC::LoadDataWithTranspose(a2Local, a1Local, loadDataParams);
        inQueueA2.EnQue<T2>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void CopyInB1(size_t SpStart)
    {
        AscendC::LocalTensor<T2> b1Local = inQueueB1.AllocTensor<T2>();
        #pragma unroll
        for (size_t id = 0; id < SPTILE; id++)
        {
            size_t i = id / COL_TILE, j = id % COL_TILE;
            size_t src_offset = colGlobal.GetValue(SpStart + id) * this->D;
            AscendC::DataCopy(b1Local[i * COL_TILE * MAXD + j * NUM_DATABLOCK_ELES], denseGlobal[src_offset], copyParams);
        }
        inQueueB1.EnQue<T2>(b1Local);
    }
    
    __aicore__ inline void CopyInB1last(size_t SpStart, size_t len)
    {
        AscendC::LocalTensor<T2> b1Local = inQueueB1.AllocTensor<T2>();
        #pragma unroll
        for (size_t id = 0; id < len; id++)
        {
            size_t i = id / COL_TILE, j = id % COL_TILE;
            size_t src_offset = colGlobal.GetValue(SpStart + id) * this->D;
            AscendC::DataCopy(b1Local[i * COL_TILE * MAXD + j * NUM_DATABLOCK_ELES], denseGlobal[src_offset], copyParams);
        }
        inQueueB1.EnQue<T2>(b1Local);
    }
    __aicore__ inline void CopyOut(size_t dst_offset)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = D;
        fixpipeParams.mSize = ROW_WINDOW;
        fixpipeParams.srcStride = ROW_WINDOW;
        fixpipeParams.dstStride = D;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr(std::is_same_v<T2, half>) fixpipeParams.quantPre = QuantMode_t::F322F16;
        // if (dst_offset==4 * this->D * ROW_WINDOW)
        // {
            // uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(n)};
            // AscendC::ShapeInfo shapeInfo(2, array);
            // DumpTensor(c1Local, 0, n * 16, shapeInfo);
            // DumpTensor(yGlobal[dst_offset], 0, n * 16, shapeInfo);
        // }
        AscendC::SetAtomicAdd<T2>();
        AscendC::Fixpipe(yGlobal[dst_offset], c1Local, fixpipeParams);
        AscendC::SetAtomicNone();
        outQueueCO1.FreeTensor(c1Local);
        // if (dst_offset==4 * this->D * ROW_WINDOW)
        // {
            // uint32_t array[] = {static_cast<uint32_t>(16),static_cast<uint32_t>(n)};
            // AscendC::ShapeInfo shapeInfo(2, array);
            // DumpTensor(c1Local, 0, n * 16, shapeInfo);
            // DumpTensor(yGlobal[dst_offset], 0, n * 16, shapeInfo);
        // }
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    size_t L, R, D, rowptr_idx, next_rowptr;
    AscendC::GlobalTensor<T1> rowptrGlobal, colGlobal, startGlobal;
    AscendC::GlobalTensor<T2> denseGlobal, yGlobal, valueGlobal;

    AscendC::DataCopyParams copyParams;
};


// #define BLOCK 1024
// template<typename T1, typename T2> class KernelSetAtomicAdd {
// public:
//     __aicore__ inline KernelSetAtomicAdd() {}
//     __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col_id, GM_ADDR block_local_id, GM_ADDR block_ptr, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, size_t size, size_t len, size_t D)
//     {
//         rowptrGlobal.SetGlobalBuffer((__gm__ T1*)rowptr);
//         colGlobal.SetGlobalBuffer((__gm__ T1*)col_id);
//         localidGlobal.SetGlobalBuffer((__gm__ int16_t*)block_local_id);
//         blockptrGlobal.SetGlobalBuffer((__gm__ T1*)block_ptr);
//         startGlobal.SetGlobalBuffer((__gm__ T1*)start);
//         valueGlobal.SetGlobalBuffer((__gm__ T2*)value);
//         denseGlobal.SetGlobalBuffer((__gm__ T2*)dense);
//         yGlobal.SetGlobalBuffer((__gm__ T2*)y);
        
//         rowptrGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         colGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         localidGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         blockptrGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         startGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         valueGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

//         _IDX.SetGlobalBuffer((__gm__ uint64_t*)block_local_id);
//         _IDX.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         _VAL.SetGlobalBuffer((__gm__ uint64_t*)value);
//         _VAL.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         _COL.SetGlobalBuffer((__gm__ uint64_t*)col_id);
//         _COL.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

//         size_t L = (GetBlockIdx() + 20) * len, R = (GetBlockIdx() + 20 + 1) * len;
//         if (L > size) L = size;
//         if (R > size) R = size;
//         this->L = L; this->R = R; this->D = D;
//         this->rowptr_idx = startGlobal.GetValue(GetBlockIdx() + 20);
//         this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
        
//         AscendC::LocalMemAllocator allocator;
//         this->yLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(ROW_WINDOW * D);
//         this->yLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(ROW_WINDOW * D);
//         // for (int i = 0; i < COL_TILE; i++) this->tempLocal0[i] = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(D);
//         // for (int i = 0; i < COL_TILE; i++) this->tempLocal1[i] = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(D);
//         this->tempLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(COL_TILE * D);
//         this->tempLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(COL_TILE * D);

//         this->colLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T1>(COL_TILE);
//         this->colLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T1>(COL_TILE);
//         this->idxLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, int16_t>(ROW_WINDOW * COL_TILE);
//         this->idxLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, int16_t>(ROW_WINDOW * COL_TILE);
//         this->valLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(ROW_WINDOW * COL_TILE);
//         this->valLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(ROW_WINDOW * COL_TILE);
//     }
//     __aicore__ inline void proc()
//     {
//         Process();
//     }
//     __aicore__ inline void Process()
//     {
//         size_t dst_offset = this->rowptr_idx * ROW_WINDOW * this->D, d, tag = 0, pingpongy = 0, pingpongc = 0;
//         size_t i, j;
//         if (L == R) return;
//                     // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
//                     // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
//             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
//             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
//             AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(3);
//         AscendC::DataCopyParams densecopyParams;
//         densecopyParams.blockCount = this->D * sizeof(T2) / 32;
//         densecopyParams.blockLen = 1;
//         densecopyParams.srcStride = 0;
//         densecopyParams.dstStride = 0;

//         size_t bl = blockptrGlobal.GetValue(L), tlen = blockptrGlobal.GetValue(L + 1) - bl;
//         AscendC::DataCopy(colLocal0, colGlobal[L * COL_TILE], COL_TILE);
//         AscendC::DataCopyParams copyParams{1, (uint16_t)(tlen * sizeof(int16_t)), 0, 0}; 
//         AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
//         AscendC::DataCopyPad(idxLocal0, localidGlobal[bl], copyParams, padParams);
//         copyParams.blockLen = tlen * sizeof(T2);
//         AscendC::DataCopyPad(valLocal0, valueGlobal[bl], copyParams, padParams);
//             AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(4);
//         for (i = L; i < R; i++)
//         {
//         // AscendC::SyncAll<false>();
//             // printf("%d %d %d %d\n",GetBlockIdx(), i,L,R);
//             if (i == this->next_rowptr)
//             {
//                 AscendC::SetAtomicAdd<T2>();
//                     //                 {
//                     //     uint32_t array[] = {static_cast<uint32_t>(ROW_WINDOW),static_cast<uint32_t>(this->D)};
//                     //     AscendC::ShapeInfo shapeInfo(2, array);
//                     //     DumpTensor(yLocal, 0, this->D * ROW_WINDOW, shapeInfo);
//                     // }
//                 auto newyLocal = pingpongy ? yLocal0 : yLocal1;
//                 auto yLocal = pingpongy ? yLocal1 : yLocal0;
//                 AscendC::Duplicate(newyLocal, (float)0, ROW_WINDOW * this->D);
//                 if constexpr(std::is_same_v<float, T2>)
//                 {
//                     AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
//                     AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
//                     AscendC::DataCopy(yGlobal[dst_offset], yLocal, ROW_WINDOW * this->D);
//                 }
//                 if constexpr(std::is_same_v<half, T2>)
//                 {
//                     auto temp = yLocal.ReinterpretCast<T2>();
//                     AscendC::Cast(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, ROW_WINDOW * this->D);
//                     AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
//                     AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
//                     AscendC::DataCopy(yGlobal[dst_offset], temp, ROW_WINDOW * this->D);
//                 }
//                 AscendC::SetAtomicNone();
//                 pingpongy ^= 1;
//                 do
//                 {
//                     this->rowptr_idx++;
//                     dst_offset += this->D * ROW_WINDOW;
//                     this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
//                 }
//                 while (i == this->next_rowptr);
//             }
//             auto yLocal = pingpongy ? yLocal1 : yLocal0;
//             auto tempLocal = pingpongc ? tempLocal1 : tempLocal0;
//             size_t block_l = blockptrGlobal.GetValue(i), block_r = blockptrGlobal.GetValue(i + 1), len = block_r - block_l, offset = i * COL_TILE;
//             // AscendC::DataCachePreload(_COL, offset * sizeof(T1));
//             // AscendC::DataCachePreload(_IDX, block_l * sizeof(int16_t));
//             // AscendC::DataCachePreload(_VAL, block_l * sizeof(T2));
//             // __builtin_prefetch(colGlobal[offset].GetPhyAddr(),0,0);
//             // T1 s1[ROW_WINDOW * COL_TILE];
//             // #pragma unroll
//             // for (size_t t = 0; t < COL_TILE; t++) s1[t] += colGlobal[offset].GetValue(t);
//             // #pragma unroll
//             // for (size_t t = block_l; t < block_r; t++)
//             // {
//             //     s1[t - block_l] += localidGlobal.GetValue(t) + (T1)valueGlobal.GetValue(t);
//             // }
            
//             if (i < R-1)
//             {
//                 auto colLocal = pingpongc ? colLocal0 : colLocal1;
//                 auto idxLocal = pingpongc ? idxLocal0 : idxLocal1;
//                 auto valLocal = pingpongc ? valLocal0 : valLocal1;
//                 size_t nblock_r = blockptrGlobal.GetValue(i + 2), tlen = nblock_r - block_r;
//                 // printf("Wait %d\n",(1 - pingpongc) + 2);

//                 AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>((1 - pingpongc) + 2);
//                 AscendC::DataCopy(colLocal, colGlobal[(i + 1) * COL_TILE], COL_TILE);
//                 AscendC::DataCopyParams copyParams{1, (uint16_t)(tlen * sizeof(int16_t)), 0, 0}; 
//                 AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
//                 AscendC::DataCopyPad(idxLocal, localidGlobal[block_r], copyParams, padParams);
//                 copyParams.blockLen = tlen * sizeof(T2);
//                 AscendC::DataCopyPad(valLocal, valueGlobal[block_r], copyParams, padParams);
//                 AscendC::SetFlag<AscendC::HardEvent::MTE2_S>((1 - pingpongc) + 4);
//                 // printf("Set %d\n",(1 - pingpongc) + 4);
//             }
            
//             auto colLocal = pingpongc ? colLocal1 : colLocal0;
//             auto idxLocal = pingpongc ? idxLocal1 : idxLocal0;
//             auto valLocal = pingpongc ? valLocal1 : valLocal0;
//             // AscendC::DataCopy(colLocal, colGlobal[i * COL_TILE], COL_TILE);
//             // AscendC::DataCopyParams copyParams{1, (uint16_t)(len * sizeof(int16_t)), 0, 0}; 
//             // AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
//             // AscendC::DataCopyPad(idxLocal, localidGlobal[block_l], copyParams, padParams);
//             // copyParams.blockLen = len * sizeof(T2);
//             // AscendC::DataCopyPad(valLocal, valueGlobal[block_l], copyParams, padParams);
//             // AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(pingpongc + 2);

//                 // printf("Wait %d\n",pingpongc + 4);
//             AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(pingpongc + 4);
//             AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpongc);
//             #pragma unroll
//             for (size_t t = 0; t < COL_TILE; t++)
//             {
//                 AscendC::DataCopy(tempLocal[t * this->D], denseGlobal[colLocal.GetValue(t) * this->D], densecopyParams);
//             }
//             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpongc);
//             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpongc);
//             #pragma unroll
//             for (size_t t = 0; t < len; t++)
//             {
//                 size_t row = idxLocal.GetValue(t) >> 4, col = idxLocal.GetValue(t) & 15;
//                 T2 v = valLocal.GetValue(t);
//                 AscendC::Axpy(yLocal[row * this->D], tempLocal[col * this->D], v, D);
//             }
//             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpongc);
//             AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(pingpongc + 2);
//                 // printf("Set %d\n",pingpongc + 2);
//             pingpongc ^= 1;
//         }
//         auto yLocal = pingpongy ? yLocal1 : yLocal0;
//             // printf("OK.\n");
//         if (L < R)
//         {
//             AscendC::SetAtomicAdd<T2>();
//             if constexpr(std::is_same_v<float, T2>)
//             {
//                 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
//                 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
//                 AscendC::DataCopy(yGlobal[dst_offset], yLocal, ROW_WINDOW * this->D);
//             }
//             if constexpr(std::is_same_v<half, T2>)
//             {
//                 auto temp = yLocal.ReinterpretCast<T2>();
//                 AscendC::Cast(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, ROW_WINDOW * this->D);
//                 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
//                 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
//                 AscendC::DataCopy(yGlobal[dst_offset], temp, ROW_WINDOW * this->D);
//             }
//             AscendC::SetAtomicNone();
//         }
//             // printf("OK.\n");
//             AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
//             AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
//             AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(2);
//             AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(3);
//             // printf("OK.\n");
//     }
// private:
//     size_t L, R, D, rowptr_idx, next_rowptr;
//     AscendC::LocalTensor<float> yLocal0, yLocal1;
//     AscendC::LocalTensor<T2> tempLocal0, tempLocal1;
//     AscendC::GlobalTensor<T1> blockptrGlobal;
//     AscendC::GlobalTensor<int16_t> localidGlobal;
//     AscendC::GlobalTensor<T1> rowptrGlobal, colGlobal, startGlobal;
//     AscendC::GlobalTensor<T2> valueGlobal, denseGlobal, yGlobal;
//     AscendC::GlobalTensor<uint64_t> _IDX, _COL, _VAL;
    
    
//     AscendC::LocalTensor<T2> valLocal0, valLocal1;
//     AscendC::LocalTensor<int16_t> idxLocal0, idxLocal1;
//     AscendC::LocalTensor<T1> colLocal0, colLocal1;
// };
template<typename T1, typename T2> class KernelSetAtomicAdd {
public:
    static_assert(sizeof(T2) == 2 || sizeof(T2) == 4, "T2 must be 2 or 4 bytes");
    static constexpr size_t MAXD = sizeof(T2) == 2 ? 1024 : 512;
    __aicore__ inline KernelSetAtomicAdd() {}
    __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, size_t size, size_t len, size_t D)
    {
        rowptrGlobal.SetGlobalBuffer((__gm__ T1*)rowptr);
        colGlobal.SetGlobalBuffer((__gm__ T1*)col);
        startGlobal.SetGlobalBuffer((__gm__ T1*)start);
        valueGlobal.SetGlobalBuffer((__gm__ T2*)value);
        denseGlobal.SetGlobalBuffer((__gm__ T2*)dense);
        yGlobal.SetGlobalBuffer((__gm__ T2*)y);

        rowptrGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // colGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        startGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // valueGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        size_t L = GetBlockIdx() * len, R = (GetBlockIdx() + 1) * len;
        if (L > size) L = size;
        if (R > size) R = size;
        this->L = L; this->R = R; this->D = D;
        this->rowptr_idx = L >= R ? 0 : startGlobal.GetValue(GetBlockIdx());
        this->next_rowptr = L >= R ? 0 : rowptrGlobal.GetValue(this->rowptr_idx + 1);
        
        AscendC::LocalMemAllocator allocator;
        this->yLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(8 * MAXD);
        // for (int i = 0; i < COL_TILE; i++) this->tempLocal0[i] = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(D);
        // for (int i = 0; i < COL_TILE; i++) this->tempLocal1[i] = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(D);
        this->tempLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(BLOCK * MAXD);
        // this->tempLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(BLOCK * MAXD);

        this->colLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T1>(MAXL);
        this->colLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T1>(MAXL);
        this->valLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(MAXL);
        this->valLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(MAXL);
        
        densecopyParams.blockCount = this->D * sizeof(T2) / 32;
        densecopyParams.blockLen = 1;
        densecopyParams.srcStride = 0;
        densecopyParams.dstStride = 0;
    }
    __aicore__ inline void ProcessLine(size_t l, size_t r, size_t dst_offset, AscendC::LocalTensor<float> &yLocal, AscendC::LocalTensor<T2> &tempLocal, size_t pingpongy)
    {
        // __ubuf__ T1* col_offset = reinterpret_cast<__ubuf__ T1*>(colLocal.address_.bufferAddr);
        // __ubuf__ T2* val_offset = reinterpret_cast<__ubuf__ T2*>(valLocal.address_.bufferAddr);
        size_t i;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongy);
        AscendC::Duplicate<float, false>(yLocal, (float)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        #pragma unroll
        for (i = l; i + 4 <= r; i += 4)
        {
            T1 col[4];
            T2 val[4];
            // col[0] = col_offset[i];
            // col[1] = col_offset[i + 1];
            // col[2] = col_offset[i + 2];
            // col[3] = col_offset[i + 3];
            // val[0] = val_offset[i];
            // val[1] = val_offset[i + 1];
            // val[2] = val_offset[i + 2];
            // val[3] = val_offset[i + 3];
            // col[4] = col_offset[i + 4];
            // col[5] = col_offset[i + 5];
            // col[6] = col_offset[i + 6];
            // col[7] = col_offset[i + 7];
            // val[4] = val_offset[i + 4];
            // val[5] = val_offset[i + 5];
            // val[6] = val_offset[i + 6];
            // val[7] = val_offset[i + 7];
            col[0] = colGlobal(i);
            col[1] = colGlobal(i + 1);
            col[2] = colGlobal(i + 2);
            col[3] = colGlobal(i + 3);
            val[0] = valueGlobal(i);
            val[1] = valueGlobal(i + 1);
            val[2] = valueGlobal(i + 2);
            val[3] = valueGlobal(i + 3);
            #define INSTRUCT(x) \
            AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col[x] * this->D], densecopyParams); \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0); \
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val[x], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); \
            cnt++; cnt &= (BLOCK - 1);

            INSTRUCT(0)
            INSTRUCT(1)
            INSTRUCT(2)
            INSTRUCT(3)
            
            // INSTRUCT(4)
            // INSTRUCT(5)
            // INSTRUCT(6)
            // INSTRUCT(7)
            // {
            //     uint32_t array[] = {static_cast<uint32_t>(this->D)};
            //     AscendC::ShapeInfo shapeInfo(1, array);
            //     DumpTensor(tempLocal[(i & 15) * MAXD], 0, this->D, shapeInfo);
            // }
        }
        for (;i < r; i++)
        {
            T1 col;
            T2 val;
            col = colGlobal(i);
            val = valueGlobal(i);
            // col = col_offset[i];
            // val = val_offset[i];
            AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col * this->D], densecopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) });
            cnt++; cnt &= (BLOCK - 1);
        }
        if constexpr(std::is_same_v<float, T2>)
        {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::DataCopy(yGlobal[dst_offset], yLocal, this->D);
        }
        if constexpr(std::is_same_v<half, T2>)
        {
            auto temp = yLocal.ReinterpretCast<T2>();
            AscendC::Cast<T2, float, false>(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 2 * sizeof(T2), 8 });
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::DataCopy(yGlobal[dst_offset], temp, this->D);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongy);
    }
    __aicore__ inline void ProcessLineMB(size_t l, size_t r, size_t dst_offset, AscendC::LocalTensor<float> &yLocal, AscendC::LocalTensor<T2> &tempLocal, size_t pingpongy)
    {
        // __ubuf__ T1* col_offset = reinterpret_cast<__ubuf__ T1*>(colLocal.address_.bufferAddr);
        // __ubuf__ T2* val_offset = reinterpret_cast<__ubuf__ T2*>(valLocal.address_.bufferAddr);
        size_t i;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpongy);
        AscendC::Duplicate<float, false>(yLocal, (float)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        #pragma unroll
        for (i = l; i + 4 <= r; i += 4)
        {
            T1 col[4];
            T2 val[4];
            // col[0] = col_offset[i];
            // col[1] = col_offset[i + 1];
            // col[2] = col_offset[i + 2];
            // col[3] = col_offset[i + 3];
            // val[0] = val_offset[i];
            // val[1] = val_offset[i + 1];
            // val[2] = val_offset[i + 2];
            // val[3] = val_offset[i + 3];
            // col[4] = col_offset[i + 4];
            // col[5] = col_offset[i + 5];
            // col[6] = col_offset[i + 6];
            // col[7] = col_offset[i + 7];
            // val[4] = val_offset[i + 4];
            // val[5] = val_offset[i + 5];
            // val[6] = val_offset[i + 6];
            // val[7] = val_offset[i + 7];
            col[0] = colGlobal(i);
            col[1] = colGlobal(i + 1);
            col[2] = colGlobal(i + 2);
            col[3] = colGlobal(i + 3);
            val[0] = valueGlobal(i);
            val[1] = valueGlobal(i + 1);
            val[2] = valueGlobal(i + 2);
            val[3] = valueGlobal(i + 3);
            #define INSTRUCT(x) \
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i & 7); \
            AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col[x] * this->D], densecopyParams); \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(i & 7); \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(i & 7); \
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val[x], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); \
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(i & 7); \
            cnt++; cnt &= (BLOCK - 1);

            INSTRUCT(0)
            INSTRUCT(1)
            INSTRUCT(2)
            INSTRUCT(3)
            
            // INSTRUCT(4)
            // INSTRUCT(5)
            // INSTRUCT(6)
            // INSTRUCT(7)
            // {
            //     uint32_t array[] = {static_cast<uint32_t>(this->D)};
            //     AscendC::ShapeInfo shapeInfo(1, array);
            //     DumpTensor(tempLocal[(i & 15) * MAXD], 0, this->D, shapeInfo);
            // }
        }
        for (;i < r; i++)
        {
            T1 col;
            T2 val;
            col = colGlobal(i);
            val = valueGlobal(i);
            // col = col_offset[i];
            // val = val_offset[i];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(i & 3);
            AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col * this->D], densecopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(i & 3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(i & 3);
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) });
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(i & 3);
            cnt++; cnt &= (BLOCK - 1);
        }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
        if constexpr(std::is_same_v<float, T2>)
        {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::DataCopy(yGlobal[dst_offset], yLocal, this->D);
        }
        if constexpr(std::is_same_v<half, T2>)
        {
            auto temp = yLocal.ReinterpretCast<T2>();
            AscendC::Cast<T2, float, false>(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 2 * sizeof(T2), 8 });
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpongy);
            AscendC::DataCopy(yGlobal[dst_offset], temp, this->D);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpongy);
    }
    __aicore__ inline void Process()
    {
        if (L == R) return;
        AscendC::SetAtomicAdd<T2>();
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T2, AscendC::MaskMode::COUNTER>(this->D);
        size_t src_offset, dst_offset = this->rowptr_idx * this->D, d, pingpongy = 0;

        // AscendC::DataCopyPad(colLocal0, colGlobal[L], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
        // AscendC::DataCopyPad(valLocal0, valueGlobal[L], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
        // AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(6);
                // AscendC::Duplicate(colLocal.template ReinterpretCast<int32_t>(), 0, MAXL * sizeof(T1) / 4);
                // AscendC::Duplicate(valLocal, (T2)0, MAXL);;
                // AscendC::Duplicate(ncolLocal.template ReinterpretCast<int32_t>(), 0, MAXL * sizeof(T1) / 4);
                // AscendC::Duplicate(nvalLocal, (T2)0, MAXL);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
        for (size_t i = L; i < R; i += MAXL)
        {
            if (i + 2 * MAXL < R)
            {
                AscendC::DataCopyPad(colLocal1, colGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
                AscendC::DataCopyPad(valLocal1, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
                // AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(7);
            }
            // AscendC::DataCopy(nvalLocal, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2) / 32), 0, 0});
            // AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(6);
            size_t SpStart = i, SpEnd = min(i + MAXL, R);
            while (SpStart < SpEnd)
            {
                auto yLocal = yLocal0[pingpongy * MAXD];
                size_t r = min(this->next_rowptr, SpEnd);
                if (SpStart < r)
                {
                    ProcessLine(SpStart, r, dst_offset, yLocal, tempLocal0, pingpongy);
                    pingpongy ++;
                    pingpongy &= 7;
                    SpStart = r;
                }
                if (r == this->next_rowptr)
                {
                    this->rowptr_idx++;
                    dst_offset += this->D;
                    this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
                }
            }
            i += MAXL;
            if (i >= R) break;
            if (i + 2 * MAXL < R)
            {
                AscendC::DataCopyPad(colLocal0, colGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
                AscendC::DataCopyPad(valLocal0, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
                // AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(6);
            }
            // AscendC::DataCopy(nvalLocal, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2) / 32), 0, 0});
            // AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(7);
            SpStart = i, SpEnd = min(i + MAXL, R);
            while (SpStart < SpEnd)
            {
                auto yLocal = yLocal0[pingpongy * MAXD];
                size_t r = min(this->next_rowptr, SpEnd);
                if (SpStart < r)
                {
                    ProcessLine(SpStart, r, dst_offset, yLocal, tempLocal0, pingpongy);
                    pingpongy ++;
                    pingpongy &= 7;
                    SpStart = r;
                }
                if (r == this->next_rowptr)
                {
                    this->rowptr_idx++;
                    dst_offset += this->D;
                    this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
                }
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::SetAtomicNone();
    }
private:
    size_t L, R, D, rowptr_idx, next_rowptr, cnt = 0;
    AscendC::LocalTensor<float> yLocal0, yLocal1;
    AscendC::LocalTensor<T2> tempLocal0, tempLocal1;
    AscendC::GlobalTensor<T1> rowptrGlobal, colGlobal, startGlobal;
    AscendC::GlobalTensor<T2> valueGlobal, denseGlobal, yGlobal;
    
    AscendC::LocalTensor<T2> valLocal0, valLocal1;
    AscendC::LocalTensor<T1> colLocal0, colLocal1;
    AscendC::DataCopyParams densecopyParams;
};
extern "C" __global__ __aicore__ void spmm_ascend_cube(GM_ADDR Crowptr, GM_ADDR Ccol_id, GM_ADDR Cvalue, GM_ADDR Vrowptr, GM_ADDR Vcol_id, GM_ADDR Vvalue, GM_ADDR dense, GM_ADDR Cstart, GM_ADDR Vstart, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    // TODO: user kernel impl
    if ASCEND_IS_AIC {
        TPipe pipe;
        KernelMmad<DTYPE_CCOL_ID, DTYPE_DENSE> op;
        op.Init(dense, y, Cvalue, Cstart, Crowptr, Ccol_id, tiling_data.Csize, tiling_data.Clen, tiling_data.D, &pipe);
        // printf("%d\n",GetBlockIdx());
        // AscendC::SyncAll<false>();
        op.Process();
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        KernelSetAtomicAdd<DTYPE_VCOL_ID, DTYPE_DENSE> op;
        op.Init(Vrowptr, Vcol_id, Vvalue, dense, Vstart, y, tiling_data.Vsize, tiling_data.Vlen, tiling_data.D);
        // AscendC::SyncAll<false>();
        op.Process();
    }
}