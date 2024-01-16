//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/BuildOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Save build configuration information to JSON.
 */
class BuildOutput final : public OutputInterface
{
  public:
    //! Category of data to write
    Category category() const final { return Category::system; };

    //! Key for the entry inside the category.
    std::string label() const final { return "build"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
