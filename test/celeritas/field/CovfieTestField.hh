//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CovfieTestField.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

#if CELERITAS_USE_COVFIE

//! Write a tiny 3D covfie field (nx x ny x nz) with Bi = index_i
void write_cart_covfie(std::string const& path,
                       std::size_t nx,
                       std::size_t ny,
                       std::size_t nz);

//! Write a tiny 2D covfie field (nr x nz) with Br = ir, Bz = iz
void write_rz_covfie(std::string const& path, std::size_t nr, std::size_t nz);

#else

inline void
write_cart_covfie(std::string const&, std::size_t, std::size_t, std::size_t)
{
    CELER_NOT_CONFIGURED("covfie");
}

inline void write_rz_covfie(std::string const&, std::size_t, std::size_t)
{
    CELER_NOT_CONFIGURED("covfie");
}

#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
