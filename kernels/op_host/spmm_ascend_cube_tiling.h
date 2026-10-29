
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SpmmAscendCubeTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, Csize);
  TILING_DATA_FIELD_DEF(size_t, Clen);
  TILING_DATA_FIELD_DEF(uint32_t, Vsize);
  TILING_DATA_FIELD_DEF(size_t, Vlen);
  TILING_DATA_FIELD_DEF(size_t, D);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SpmmAscendCube, SpmmAscendCubeTilingData)
}
