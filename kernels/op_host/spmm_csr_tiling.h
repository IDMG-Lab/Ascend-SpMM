
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SpmmCsrTilingData)
  TILING_DATA_FIELD_DEF(size_t, size);
  TILING_DATA_FIELD_DEF(size_t, len);
  TILING_DATA_FIELD_DEF(size_t, D);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SpmmCsr, SpmmCsrTilingData)
}
