//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.hh
//---------------------------------------------------------------------------//

#include "celeritas/field/CartMapFieldInput.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! Run on device and return results
void field_test(CartMapFieldInput&, Span<real_type>&, Real3&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
