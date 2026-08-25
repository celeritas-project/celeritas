//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <map>
#include <memory>
#include <string>

#include "corecel/cont/EnumArray.hh"
#include "corecel/io/FileOrConsole.hh"

#include "OutputInterface.hh"

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
 *
 * \par Newline-delimited json output
 * The output registry will avoid newlines in its output by default
 * (when `os.width() == 0` or using ), allowing compatibility with NDJSON/
 JSONL :
 * \code
 * std::ofstream out("foo.jsonl");
 * out << reg << std::endl;
 * out << reg << std::endl;
 * \endcode
 */
class OutputRegistry
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstInterface = std::shared_ptr<OutputInterface const>;
    //!@}

    enum class OpenMode
    {
        app,  //!< Seek to end of file before each write (default)
        trunc,  //!< Overwrite existing file
    };

  public:
    //// PERSISTENT OUTPUT FILE MANAGEMENT ////

    // Append to a persistent JSONL file for writing with `output`
    void open(std::string s);

    // Append to a persistent JSONL file for writing with `output`
    void open(std::string s, OpenMode);

    //! Close persistent file if open
    void close() { outf_ = nullptr; }

    // Get persistent output filename (error if not open)
    std::string const& output_filename() const;

    // Append a line of JSON output to the persistent output file
    void output() const;

    // Get output filename if open

    //! Whether a persistent file is open
    bool is_open() const { return static_cast<bool>(outf_); }

    //// INTERFACE MANAGEMENT ////

    // Add an interface for writing
    void insert(SPConstInterface);

    // Write output to the given JSON object
    void output(JsonPimpl*) const;

    // Dump all outputs as JSON to the given stream
    void output(std::ostream* os) const;

    // Whether no output has been registered
    bool empty() const;

    //! Output to a stream
    friend std::ostream& operator<<(std::ostream& os, OutputRegistry const& reg)
    {
        reg.output(&os);
        return os;
    }

  private:
    using Category = OutputInterface::Category;

    struct FOSDeleter
    {
        void operator()(FileOrStdout*) const;
    };

    // Output file or stdout
    std::unique_ptr<FileOrStdout, FOSDeleter> outf_;

    // Interfaces by category
    EnumArray<Category, std::map<std::string, SPConstInterface>> interfaces_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Add an interfaces for writing system diagnostics
void insert_system_diagnostics(OutputRegistry&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
