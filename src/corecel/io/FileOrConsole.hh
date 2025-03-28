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
    explicit FileOrStdin(std::string filename);

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
 * It includes a facility to writing a unique output filename.
 */
class FileOrStdout
{
  public:
    //! How to open the file if not writing to stdout
    enum class OpenMode
    {
        overwrite,  //!< Replace an existing file silently
        error_if_exists,  //!< Throw an error if it exists
        unique,  //!< Generate a unique replacement
        size_
    };

  public:
    // Construct with a filename and the default modee
    FileOrStdout(std::string filename, OpenMode mode);

    //! Construct with a filename and default to overwrite if not stdout
    explicit FileOrStdout(std::string filename)
        : FileOrStdout{std::move(filename), OpenMode::overwrite}
    {
    }

    //! Implicitly cast to the opened stream
    operator std::ostream&() { return outf_.is_open() ? outf_ : std::cout; }

    //! Implicitly cast as a pointer to the opened stream
    operator std::ostream*() { return &static_cast<std::ostream&>(*this); }

    //! Get the filename or renamed placeholder
    std::string const& filename() const& { return filename_; }

    //! Get the filename or renamed placeholder (during move)
    std::string&& filename() && { return std::move(filename_); }

    //! Whether we're writing to cout
    bool is_stdout() const { return !outf_.is_open(); }

  private:
    std::string filename_;
    std::ofstream outf_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
