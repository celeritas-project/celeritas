//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
struct JsonPimpl;

//---------------------------------------------------------------------------//
/*!
 * Pure abstract interface for writing metadata output to JSON.
 *
 * At the end of the program/run, the OutputRegistry will call the "output"
 * method on all interfaces.
 *
 * \todo Perhaps another output method for saving a schema?
 */
class OutputInterface
{
  public:
    //! Output category (TODO: could replace with string/cstring?)
    enum class Category
    {
        input,
        result,
        system,
        internal,
        size_
    };

  public:
    //! Category of data to write
    virtual Category category() const = 0;

    //! Key for the entry inside the category.
    virtual std::string label() const = 0;

    // Write output to the given JSON object
    virtual void output(JsonPimpl*) const = 0;

  protected:
    // Protected destructor prevents direct deletion of pointer-to-interface
    ~OutputInterface() = default;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get a string representation of the category
char const* to_cstring(OutputInterface::Category value);

// Get the JSON representation of a single output (mostly for testing)
std::string to_string(OutputInterface const& output);

//---------------------------------------------------------------------------//
}  // namespace celeritas
