//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OpenmpOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Save OpenMP environment/config information to JSON.
 *
 * This saves maximum number of threads as well as binding information. Since
 * this should never be run inside a \c parallel block, the local \c
 * num_threads value is never saved.
 */
class OpenmpOutput final : public OutputInterface
{
  public:
    //! Category of data to write
    Category category() const final { return Category::system; };

    //! Key for the entry inside the category.
    std::string_view label() const final { return "openmp"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
