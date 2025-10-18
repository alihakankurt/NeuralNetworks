public sealed class IncompatibleShapeException : Exception
{
    public IncompatibleShapeException(string argName1, string argName2) : base($"{argName1} and {argName2} have incompatible shapes to work with.")
    {
    }
}
