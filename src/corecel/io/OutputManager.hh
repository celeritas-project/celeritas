//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <map>
#include <memory>
#include <string>

#include "OutputInterface.hh"
#include "corecel/cont/EnumArray.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store classes that can output data at the end of the run.
 *
 * Each output interface defines a category (e.g. input, result, system) and a
 * name. The output manager then writes the JSON output from that entry into a
 * nested database:
 * \verbatim
   {"category": {"label": "data"}}
 * \endverbatim
 */
class OutputManager
{
  public:
    //!@{
    //! Type aliases
    using SPConstInterface = std::shared_ptr<const OutputInterface>;
    //!@}

  public:
    // Add an interface for writing
    void insert(SPConstInterface);

    // Write all outputs as JSON to the given stream
    void output(std::ostream* os) const;

  private:
    using Category = OutputInterface::Category;

    // Interfaces by category
    EnumArray<Category, std::map<std::string, SPConstInterface>> interfaces_;
};

// TODO: potentially add a global instance here?

//---------------------------------------------------------------------------//
} // namespace celeritas
