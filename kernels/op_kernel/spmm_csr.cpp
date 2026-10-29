#include "kernel_operator.h"


using namespace AscendC;

#define size_t uint32_t
#define BLOCK 64
#define MAXL 1024
// template<typename T1, typename T2> class KernelSetAtomicAdd {
// public:
//     __aicore__ inline KernelSetAtomicAdd() {}
//     __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, size_t size, size_t len, size_t D)
//     {
//         rowptrGlobal.SetGlobalBuffer((__gm__ T1*)rowptr);
//         colGlobal.SetGlobalBuffer((__gm__ T1*)col);
//         startGlobal.SetGlobalBuffer((__gm__ T1*)start);
//         valueGlobal.SetGlobalBuffer((__gm__ T2*)value);
//         denseGlobal.SetGlobalBuffer((__gm__ T2*)dense);
//         yGlobal.SetGlobalBuffer((__gm__ T2*)y);

//         rowptrGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         // colGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         startGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
//         // valueGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

//         size_t L = GetBlockIdx() * len, R = (GetBlockIdx() + 1) * len;
//         if (L > size) L = size;
//         if (R > size) R = size;
//         this->L = L; this->R = R; this->D = D;
//         this->rowptr_idx = startGlobal.GetValue(GetBlockIdx());
//         this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
        
//         AscendC::LocalMemAllocator allocator;
//         this->yLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(BLOCK * MAXD);
//         // for (int i = 0; i < COL_TILE; i++) this->tempLocal0[i] = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(D);
//         // for (int i = 0; i < COL_TILE; i++) this->tempLocal1[i] = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(D);
//         this->tempLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(BLOCK * MAXD);
//         // this->tempLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(BLOCK * MAXD);

//         this->colLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T1>(MAXL);
//         this->colLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T1>(MAXL);
//         this->valLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(MAXL);
//         this->valLocal1 = allocator.Alloc<AscendC::TPosition::VECCALC, T2>(MAXL);
        
//         densecopyParams.blockCount = this->D * sizeof(T2) / 32;
//         densecopyParams.blockLen = 1;
//         densecopyParams.srcStride = 0;
//         densecopyParams.dstStride = 0;
//     }
//     __aicore__ inline void ProcessLine(size_t l, size_t r, size_t dst_offset, AscendC::LocalTensor<float> &yLocal, AscendC::LocalTensor<T2> &tempLocal, AscendC::LocalTensor<T1> &colLocal, AscendC::LocalTensor<T2> &valLocal)
//     {
//         __ubuf__ T1* col_offset = reinterpret_cast<__ubuf__ T1*>(colLocal.address_.bufferAddr);
//         __ubuf__ T2* val_offset = reinterpret_cast<__ubuf__ T2*>(valLocal.address_.bufferAddr);
//         size_t i;
//         #pragma unroll
//         for (i = l; i + 4 <= r; i += 4)
//         {
//             T1 col[4];
//             T2 val[4];
//             // col[0] = col_offset[i];
//             // col[1] = col_offset[i + 1];
//             // col[2] = col_offset[i + 2];
//             // col[3] = col_offset[i + 3];
//             // val[0] = val_offset[i];
//             // val[1] = val_offset[i + 1];
//             // val[2] = val_offset[i + 2];
//             // val[3] = val_offset[i + 3];
//             // col[4] = col_offset[i + 4];
//             // col[5] = col_offset[i + 5];
//             // col[6] = col_offset[i + 6];
//             // col[7] = col_offset[i + 7];
//             // val[4] = val_offset[i + 4];
//             // val[5] = val_offset[i + 5];
//             // val[6] = val_offset[i + 6];
//             // val[7] = val_offset[i + 7];
//             col[0] = colGlobal(i);
//             col[1] = colGlobal(i + 1);
//             col[2] = colGlobal(i + 2);
//             col[3] = colGlobal(i + 3);
//             val[0] = valueGlobal(i);
//             val[1] = valueGlobal(i + 1);
//             val[2] = valueGlobal(i + 2);
//             val[3] = valueGlobal(i + 3);
//             #define INSTRUCT(x) \
//             AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col[x] * this->D], densecopyParams); \
//             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(x); \
//             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(x); \
//             AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val[x], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); \
//             cnt++; cnt &= 31;

//             INSTRUCT(0)
//             INSTRUCT(1)
//             INSTRUCT(2)
//             INSTRUCT(3)
            
//             // INSTRUCT(4)
//             // INSTRUCT(5)
//             // INSTRUCT(6)
//             // INSTRUCT(7)
//             // {
//             //     uint32_t array[] = {static_cast<uint32_t>(this->D)};
//             //     AscendC::ShapeInfo shapeInfo(1, array);
//             //     DumpTensor(tempLocal[(i & 15) * MAXD], 0, this->D, shapeInfo);
//             // }
//         }
//         for (;i < r; i++)
//         {
//             T1 col;
//             T2 val;
//             col = colGlobal(i);
//             val = valueGlobal(i);
//             // col = col_offset[i];
//             // val = val_offset[i];
//             AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col * this->D], densecopyParams);
//             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(i & 3);
//             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(i & 3);
//             AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) });
//             cnt++; cnt &= 31;
//         }
//         if constexpr(std::is_same_v<float, T2>)
//         {
//             AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
//             AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
//             AscendC::DataCopy(yGlobal[dst_offset], yLocal, this->D);
//         }
//         if constexpr(std::is_same_v<half, T2>)
//         {
//             auto temp = yLocal.ReinterpretCast<T2>();
//             AscendC::Cast<T2, float, false>(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 2 * sizeof(T2), 8 });
//             AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
//             AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
//             AscendC::DataCopy(yGlobal[dst_offset], temp, this->D);
//         }
//     }
//     __aicore__ inline void Process()
//     {
//         AscendC::SetAtomicAdd<T2>();
//         AscendC::SetMaskCount();
//         AscendC::SetVectorMask<T2, AscendC::MaskMode::COUNTER>(this->D);
//         size_t src_offset, dst_offset = this->rowptr_idx * this->D, d, pingpongy = 0;

//         AscendC::DataCopyPad(colLocal0, colGlobal[L], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
//         AscendC::DataCopyPad(valLocal0, valueGlobal[L], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
//         AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(6);
//                 // AscendC::Duplicate(colLocal.template ReinterpretCast<int32_t>(), 0, MAXL * sizeof(T1) / 4);
//                 // AscendC::Duplicate(valLocal, (T2)0, MAXL);;
//                 // AscendC::Duplicate(ncolLocal.template ReinterpretCast<int32_t>(), 0, MAXL * sizeof(T1) / 4);
//                 // AscendC::Duplicate(nvalLocal, (T2)0, MAXL);
//         for (size_t i = L; i < R; i += MAXL)
//         {
//             if (i + MAXL < R)
//             {
//                 AscendC::DataCopyPad(colLocal1, colGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
//                 AscendC::DataCopyPad(valLocal1, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
//                 AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(7);
//             }
//             // AscendC::DataCopy(nvalLocal, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2) / 32), 0, 0});
//             AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(6);
//             size_t SpStart = i, SpEnd = min(i + MAXL, R);
//             while (SpStart < SpEnd)
//             {
//                 auto yLocal = yLocal0[pingpongy * MAXD];
//                 AscendC::Duplicate<float, false>(yLocal, (float)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
//                 size_t r = min(this->next_rowptr, SpEnd);
//                 if (SpStart < r)
//                 {
//                     ProcessLine(SpStart, r, dst_offset, yLocal, tempLocal0, colLocal0, valLocal0);
//                     pingpongy ++;
//                     pingpongy &= 31;
//                     SpStart = r;
//                 }
//                 if (r == this->next_rowptr)
//                 {
//                     this->rowptr_idx++;
//                     dst_offset += this->D;
//                     this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
//                 }
//             }
//             i += MAXL;
//             if (i >= R) break;
//             if (i + MAXL < R)
//             {
//                 AscendC::DataCopyPad(colLocal0, colGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
//                 AscendC::DataCopyPad(valLocal0, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
//                 AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(6);
//             }
//             // AscendC::DataCopy(nvalLocal, valueGlobal[i + MAXL], {1, (uint16_t)(MAXL * sizeof(T2) / 32), 0, 0});
//             AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(7);
//             SpStart = i, SpEnd = min(i + MAXL, R);
//             while (SpStart < SpEnd)
//             {
//                 auto yLocal = yLocal0[pingpongy * MAXD];
//                 AscendC::Duplicate<float, false>(yLocal, (float)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
//                 size_t r = min(this->next_rowptr, SpEnd);
//                 if (SpStart < r)
//                 {
//                     ProcessLine(SpStart, r, dst_offset, yLocal, tempLocal0, colLocal1, valLocal1);
//                     pingpongy ++;
//                     pingpongy &= 31;
//                     SpStart = r;
//                 }
//                 if (r == this->next_rowptr)
//                 {
//                     this->rowptr_idx++;
//                     dst_offset += this->D;
//                     this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
//                 }
//             }
//         }
//         AscendC::SetMaskNorm();
//         AscendC::ResetMask();
//         AscendC::SetAtomicNone();
//     }
// private:
//     size_t L, R, D, rowptr_idx, next_rowptr, cnt = 0;
//     AscendC::LocalTensor<float> yLocal0, yLocal1;
//     AscendC::LocalTensor<T2> tempLocal0, tempLocal1;
//     AscendC::GlobalTensor<T1> rowptrGlobal, colGlobal, startGlobal;
//     AscendC::GlobalTensor<T2> valueGlobal, denseGlobal, yGlobal;
    
//     AscendC::LocalTensor<T2> valLocal0, valLocal1;
//     AscendC::LocalTensor<T1> colLocal0, colLocal1;
//     AscendC::DataCopyParams densecopyParams;
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
        this->rowptr_idx = startGlobal.GetValue(GetBlockIdx());
        this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
        
        AscendC::LocalMemAllocator allocator;
        this->yLocal0 = allocator.Alloc<AscendC::TPosition::VECCALC, float>(64 * MAXD);
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
    __aicore__ inline void ProcessLine(size_t l, size_t r, size_t dst_offset, AscendC::LocalTensor<float> &yLocal, AscendC::LocalTensor<T2> &tempLocal, AscendC::LocalTensor<T1> &colLocal, AscendC::LocalTensor<T2> &valLocal)
    {
        __ubuf__ T1* col_offset = reinterpret_cast<__ubuf__ T1*>(colLocal.address_.bufferAddr);
        __ubuf__ T2* val_offset = reinterpret_cast<__ubuf__ T2*>(valLocal.address_.bufferAddr);
        size_t i;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
        int flg = 0;
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
            // #define INSTRUCT(x) 
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(flg); 
            AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col[0] * this->D], densecopyParams); 
            AscendC::DataCopy(tempLocal[(cnt + 1) * MAXD], denseGlobal[col[1] * this->D], densecopyParams); 
            AscendC::DataCopy(tempLocal[(cnt + 2) * MAXD], denseGlobal[col[2] * this->D], densecopyParams); 
            AscendC::DataCopy(tempLocal[(cnt + 3) * MAXD], denseGlobal[col[3] * this->D], densecopyParams); 
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(flg); 
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(flg); 
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[cnt * MAXD], val[0], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); 
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[(cnt + 1) * MAXD], val[1], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); 
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[(cnt + 2) * MAXD], val[2], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); 
            AscendC::Axpy<float, T2, false>(yLocal, tempLocal[(cnt + 3) * MAXD], val[3], AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 2 * sizeof(T2) }); 
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(flg); 
            cnt+=4; cnt &= 31;
            flg++;flg&=7;

            // INSTRUCT(0)
            // INSTRUCT(1)
            // INSTRUCT(2)
            // INSTRUCT(3)
            
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
            cnt++; cnt &= 31;
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
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
            AscendC::DataCopy(yGlobal[dst_offset], yLocal, this->D);
        }
        if constexpr(std::is_same_v<half, T2>)
        {
            auto temp = yLocal.ReinterpretCast<T2>();
            AscendC::Cast<T2, float, false>(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 2 * sizeof(T2), 8 });
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
            AscendC::DataCopy(yGlobal[dst_offset], temp, this->D);
        }
    }

    __aicore__ void ProcessLineDetermine(size_t l, size_t r, size_t dst_offset, AscendC::LocalTensor<float> &yLocal, AscendC::LocalTensor<T2> &tempLocal, AscendC::LocalTensor<T1> &colLocal, AscendC::LocalTensor<T2> &valLocal)
    {
        // __ubuf__ T1* col_offset = reinterpret_cast<__ubuf__ T1*>(colLocal.address_.bufferAddr);
        // __ubuf__ T2* val_offset = reinterpret_cast<__ubuf__ T2*>(valLocal.address_.bufferAddr);
        size_t i;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Duplicate(yLocal, (float)0, this->D);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (i = l; i < r; i++)
        {
            T1 col;
            T2 val;
            col = colGlobal(i);
            val = valueGlobal(i);
            // col = col_offset[i];
            // val = val_offset[i];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::DataCopy(tempLocal[cnt * MAXD], denseGlobal[col * this->D], this->D);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Axpy(yLocal, tempLocal[cnt * MAXD], val, this->D);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            cnt++; cnt &= (BLOCK - 1);
        }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        if constexpr(std::is_same_v<float, T2>)
        {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(yGlobal[dst_offset], yLocal, this->D);
        }
        if constexpr(std::is_same_v<half, T2>)
        {
            auto temp = yLocal.ReinterpretCast<T2>();
            AscendC::Cast(temp, yLocal, AscendC::RoundMode::CAST_FLOOR, this->D);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopy(yGlobal[dst_offset], temp, this->D);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }
    // __aicore__ inline void Process()
    // {
    //     AscendC::SetAtomicAdd<T2>();
    //     size_t src_offset, dst_offset = this->rowptr_idx * this->D, d, pingpongy = 0;
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    //     for (size_t i = L; i < R; i ++)
    //     {
    //         while (i == this->next_rowptr)
    //         {
    //             this->rowptr_idx++;
    //             dst_offset += this->D;
    //             this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
    //         }
    //         T1 col = colGlobal.GetValue(i);
    //         T2 v = valueGlobal.GetValue(i);
    //         auto temp = tempLocal0;
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    //         AscendC::DataCopy(temp[0 * MAXD], denseGlobal[col * this->D], this->D);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    //         AscendC::Muls(temp[0 * MAXD], temp[0 * MAXD], v, this->D);
    //         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    //         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    //         AscendC::DataCopy(yGlobal[dst_offset], temp[(0) * MAXD], this->D);
    //         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    //     }
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    //         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    //     AscendC::SetAtomicNone();
    // }
    
    __aicore__ inline void Process()
    {
        if (L == R) return;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetAtomicAdd<T2>();
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T2, AscendC::MaskMode::COUNTER>(this->D);
        size_t src_offset, dst_offset = this->rowptr_idx * this->D, d, pingpongy = 0;

        AscendC::DataCopyPad(colLocal0, colGlobal[L], {1, (uint16_t)(MAXL * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
        AscendC::DataCopyPad(valLocal0, valueGlobal[L], {1, (uint16_t)(MAXL * sizeof(T2)), 0, 0}, {false, 0, 0, 0});
        // AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(6);
                // AscendC::Duplicate(colLocal.template ReinterpretCast<int32_t>(), 0, MAXL * sizeof(T1) / 4);
                // AscendC::Duplicate(valLocal, (T2)0, MAXL);;
                // AscendC::Duplicate(ncolLocal.template ReinterpretCast<int32_t>(), 0, MAXL * sizeof(T1) / 4);
                // AscendC::Duplicate(nvalLocal, (T2)0, MAXL);
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
                AscendC::Duplicate<float, false>(yLocal, (float)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
                size_t r = min(this->next_rowptr, SpEnd);
                if (SpStart < r)
                {
                    ProcessLineDetermine(SpStart, r, dst_offset, yLocal, tempLocal0, colLocal0, valLocal0);
                    pingpongy ++;
                    pingpongy &= 63;
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
                AscendC::Duplicate<float, false>(yLocal, (float)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
                size_t r = min(this->next_rowptr, SpEnd);
                if (SpStart < r)
                {
                    ProcessLineDetermine(SpStart, r, dst_offset, yLocal, tempLocal0, colLocal1, valLocal1);
                    pingpongy ++;
                    pingpongy &= 63;
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
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::SetAtomicNone();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
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
extern "C" __global__ __aicore__ void spmm_csr(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    // TODO: user kernel impl
    KernelSetAtomicAdd<DTYPE_COL, DTYPE_DENSE> op;
    op.Init(rowptr, col, value, dense, start, y, tiling_data.size, tiling_data.len, tiling_data.D);
    op.Process();
}