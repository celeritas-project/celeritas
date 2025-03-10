//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/FileOrConsole.hh
//---------------------------------------------------------------------------//
#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct an input from an existing file, or stdin if the filename is "-".
 */
class FileOrStdin
{
  public:
    // Construct with a filename
    explicit inline FileOrStdin(std::string filename);

    //! Implicitly cast to the opened stream
    operator std::istream&() const { return *instream_; }

    //! Get the filename or renamed placeholder
    std::string const& filename() const { return filename_; }

  private:
    std::string filename_;
    std::ifstream infile_;
    std::istream* instream_{nullptr};
};

//---------------------------------------------------------------------------//
/*!
 * Construct an output to a new file, or stdout if the filename is "-".
 */
class FileOrStdout
{
  public:
    // Construct with a filename
    explicit inline FileOrStdout(std::string filename);

    //! Implicitly cast to the opened stream
    operator std::ostream&() const { return *outstream_; }

    //! Get the filename or renamed placeholder
    std::string const& filename() const { return filename_; }

  private:
    std::string filename_;
    std::ofstream outfile_;
    std::ostream* outstream_{nullptr};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with filename.
 */
FileOrStdin::FileOrStdin(std::string filename) : filename_{std::move(filename)}
{
    CELER_VALIDATE(!filename_.empty(),
                   << "empty filename is not valid for input");
    if (filename_ == "-")
    {
        instream_ = &std::cin;
        filename_ = "<stdin>";
        return;
    }
    // Open the specified file
    infile_.open(filename_);
    CELER_VALIDATE(infile_, << "failed to open '" << filename_ << "'");
    instream_ = &infile_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with filename.
 */
FileOrStdout::FileOrStdout(std::string filename)
    : filename_{std::move(filename)}
{
    CELER_VALIDATE(!filename_.empty(),
                   << "empty filename is not valid for output");
    if (filename_ == "-")
    {
        outstream_ = &std::cout;
        filename_ = "<stdout>";
        return;
    }

    // Open the specified file
    outfile_.open(filename_);
    CELER_VALIDATE(outfile_, << "failed to open '" << filename_ << "'");
    outstream_ = &outfile_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas