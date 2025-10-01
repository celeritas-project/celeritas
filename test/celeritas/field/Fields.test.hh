//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.hh
//---------------------------------------------------------------------------//

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/field/CartMapFieldInput.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! Run on device and return results
void field_test(CartMapFieldInput&, Span<real_type>&, Array<size_type, 3>&);

#if !CELER_USE_DEVICE
inline void
field_test(CartMapFieldInput&, Span<real_type>&, Array<size_type, 3>&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
