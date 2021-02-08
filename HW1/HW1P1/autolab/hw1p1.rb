require "AssessmentBase.rb"

module Hw1p1
  include AssessmentBase

  def assessmentInitialize(course)
    super("hw1p1",course)
    @problems = []
  end

end
