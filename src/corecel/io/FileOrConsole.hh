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
    operator std::istream&() { return inf_.is_open() ? inf_ : std::cin; }

    //! Get the filename or renamed placeholder
    std::string const& filename() const { return filename_; }

  private:
    std::string filename_;
    std::ifstream inf_;
};

//---------------------------------------------------------------------------//
/*!
 * Construct an output to a new file, or stdout if the filename is "-".
 *
 * \note This \em appends to the file by default to be consistent across
 * console/file, compatible with NDJSON/JSONL-style record output.
 * Construct with the \c std::ios::out mode to clobber an existing file.
 */
class FileOrStdout
{
  public:
    using Mode = std::ios::openmode;

  public:
    // Construct with a filename in "append" mode
    explicit inline FileOrStdout(std::string filename);

    // Construct with a filename with a given ios mode
    inline FileOrStdout(std::string filename, Mode);

    //! Implicitly cast to the opened stream
    operator std::ostream&() { return outf_.is_open() ? outf_ : std::cout; }

    //! Get the filename or renamed placeholder
    std::string const& filename() const { return filename_; }

  private:
    std::string filename_;
    std::ofstream outf_;
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
        filename_ = "<stdin>";
        return;
    }
    // Open the specified file
    inf_.open(filename_);
    CELER_VALIDATE(inf_, << "failed to open '" << filename_ << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with filename in "append" mode.
 *
 * This gives consistent behavior between output files and the console.
 */
FileOrStdout::FileOrStdout(std::string filename)
    : FileOrStdout{std::move(filename), std::ios::app | std::ios::out}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with filename and custom mode.
 */
FileOrStdout::FileOrStdout(std::string filename, Mode mode)
    : filename_{std::move(filename)}
{
    CELER_VALIDATE(!filename_.empty(),
                   << "empty filename is not valid for output");
    if (filename_ == "-")
    {
        CELER_VALIDATE(!(mode & std::ios::trunc), << "cannot truncate stdout");
        filename_ = "<stdout>";
        return;
    }

    // Open the specified file
    outf_.open(filename_, mode);
    CELER_VALIDATE(outf_, << "failed to open '" << filename_ << "'");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
