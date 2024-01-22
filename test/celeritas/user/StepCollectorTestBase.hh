//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollectorTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"
#include "celeritas/phys/Primary.hh"

#include "../GlobalTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class StepCollectorTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    using VecString = std::vector<std::string>;
    //!@}

  public:
    virtual VecPrimary make_primaries(size_type count) = 0;

  protected:
    template<MemSpace M>
    void run_impl(size_type num_tracks, size_type num_steps);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
