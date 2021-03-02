//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ModelIdGenerator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for constructing model IDs.
 */
class ModelIdGenerator
{
  public:
    //! Get the next model ID
    ModelId operator()() { return ModelId{id_++}; }

  private:
    ModelId::size_type id_{0};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
