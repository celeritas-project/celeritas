//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct vector on allocation.
 */
template<class T>
Pool<T, Ownership::value, MemSpace::host>::Pool()
{
    data_ = std::make_shared<std::vector<T>>();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
