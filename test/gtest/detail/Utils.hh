//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.hh
//---------------------------------------------------------------------------//
#ifndef test_detail_Utils_hh
#define test_detail_Utils_hh

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Get the "skip" message for the skip macro
const char* skip_cstring();

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#endif // test_detail_Utils_hh
