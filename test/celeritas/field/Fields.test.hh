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
#include "celeritas/field/RZMapFieldInput.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! Run CartMapField on device and return results
void field_test(CartMapFieldInput&, Span<real_type>&, Array<size_type, 3>&);

//! Run RZMapField on device and return results
void rzfield_test(RZMapFieldInput const&, Span<Real3 const>, Span<real_type>);

#if !CELER_USE_DEVICE
inline void
field_test(CartMapFieldInput&, Span<real_type>&, Array<size_type, 3>&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void
rzfield_test(RZMapFieldInput const&, Span<Real3 const>, Span<real_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
