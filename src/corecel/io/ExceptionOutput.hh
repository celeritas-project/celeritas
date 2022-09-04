//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ExceptionOutput.hh
//---------------------------------------------------------------------------//
#pragma once

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
    catch (const std::exception& e)
    {
        output_mgr.insert(std::make_shared<ExceptionOutput>(e));
    }
   \endcode
 */
class ExceptionOutput final : public OutputInterface
{
  public:
    // Construct with an exception object
    explicit ExceptionOutput(const std::exception& e);

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
} // namespace celeritas
