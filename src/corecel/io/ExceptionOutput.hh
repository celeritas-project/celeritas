//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ExceptionOutput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <memory>
#include <string>

#include "corecel/sys/TypeDemangler.hh"

#include "OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Save an exception to the 'result' for diagnostic output.
 *
 * \code
    try
    {
        ...
    }
    catch (...)
    {
        output_mgr.insert(std::make_shared<ExceptionOutput>(
            std::current_exception()));
    }
   \endcode
 */
class ExceptionOutput final : public OutputInterface
{
  public:
    // Construct with an exception pointer
    explicit ExceptionOutput(std::exception_ptr e);

    // Protected destructor
    ~ExceptionOutput();

    // Category of data to write
    Category category() const final { return Category::result; }

    // Key for the entry inside the category.
    std::string label() const final { return "exception"; }

    // Write output to the given JSON object
    void output(JsonPimpl*) const final;

  private:
    std::unique_ptr<JsonPimpl> output_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
